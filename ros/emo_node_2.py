import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
from collections import deque
import time
import torch
import cv2
import mediapipe as mp
#from emotion_pkg.model import Fullmodel42, FullmodelE2E
import wave
import subprocess
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet18
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__('emotion_recognition_node')
        self.bridge = CvBridge()
        
        # 1. Load Parameters (Now includes thresholds)
        self._load_parameters()
        
        # 2. Hardware / AI Setup
        self._setup_model()
        self._setup_mediapipe()
        self._setup_recording()
        
        # 3. State & Buffers
        self.audio_buffer = deque(maxlen=self.audio_buffer_samples)
        self.video_buffer = deque(maxlen=self.video_buffer_maxlen)
        self.ema_probs = None
        self.ema_alpha = 0.4
        self.current_emotion = "Neutral"
        self.latest_prediction_string = "Initializing..."
        self.EMOTION_MAP = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad'}

        # 4. ROS Interfaces
        self.video_sub = self.create_subscription(
            Image, self.video_topic, self.video_callback, 10)
        self.audio_sub = self.create_subscription(
            Float32MultiArray, self.audio_topic, self.audio_callback, 10)
        self.emotion_pub = self.create_publisher(
            String, self.emotion_topic, 10)
        
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("Emotion Recognition Node is fully initialized and spinning.")

    def _load_parameters(self):
        """Declares and fetches all configurable parameters from the ROS 2 parameter server."""
        # Standard configs
        self.declare_parameter('video_topic', '/camera/color/image_raw')
        self.declare_parameter('audio_topic', '/microphone/audio_data')
        self.declare_parameter('emotion_topic', '/robot/detected_emotion')
        self.declare_parameter('model_size', 'small')  
        self.declare_parameter('weights_path', '')     
        self.declare_parameter('record', False)
        self.declare_parameter('output_dir', '/tmp')
        
        # Modality Flags
        self.declare_parameter('enable_audio', True)
        self.declare_parameter('enable_video', True)
        
        # NEW: Configurable Thresholds
        self.declare_parameter('audio_energy_threshold', 0.01)
        self.declare_parameter('face_visibility_threshold', 0.5)
        self.declare_parameter('confidence_threshold', 0.5)
        
        # Fetch standard variables
        self.video_topic = self.get_parameter('video_topic').get_parameter_value().string_value
        self.audio_topic = self.get_parameter('audio_topic').get_parameter_value().string_value
        self.emotion_topic = self.get_parameter('emotion_topic').get_parameter_value().string_value
        self.weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.record_output = self.get_parameter('record').get_parameter_value().bool_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        
        self.enable_audio = self.get_parameter('enable_audio').get_parameter_value().bool_value
        self.enable_video = self.get_parameter('enable_video').get_parameter_value().bool_value
        
        # Fetch threshold variables (ROS 2 uses 'double_value' for Python floats)
        self.audio_energy_threshold = self.get_parameter('audio_energy_threshold').get_parameter_value().double_value
        self.face_visibility_threshold = self.get_parameter('face_visibility_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # Validate Model Size
        model_size = self.get_parameter('model_size').get_parameter_value().string_value.lower()
        if model_size not in ('large', 'small'):
            self.get_logger().warn(f"Invalid model_size '{model_size}'. Defaulting to 'small'.")
            model_size = 'small'
            
        self.model_size = model_size
        self.window_seconds = 3.0 if self.model_size == 'large' else 1.0
        self.num_video_frames = 16 if self.model_size == 'large' else 8
        
        # Audio/Video processing constants
        self.audio_sample_rate = 16000
        self.audio_buffer_samples = int(self.audio_sample_rate * self.window_seconds)
        self.video_assumed_fps = 40
        self.video_buffer_maxlen = int(self.video_assumed_fps * self.window_seconds)
        self.video_cleanup_margin_seconds = 0.5

    def _setup_model(self):
        """Initializes PyTorch, loads weights, and assigns the device."""
        if not os.path.exists(self.weights_path):
            self.get_logger().error(f"FATAL: Weights file not found at {self.weights_path}")
            raise SystemExit

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        self.get_logger().info("Waking up the Neural Network...")
        
        try:
            self.model = Fullmodel42(dim=256)
            self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.model.to(self.device)
            
            device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'
            self.get_logger().info(
                f"Neural Network loaded on {device_name}. "
                f"Size: {self.model_size}, Audio: {self.enable_audio}, Video: {self.enable_video}"
            )
        except Exception as e:
            self.get_logger().error(f"FATAL: Could not load the model. Error: {e}")
            raise SystemExit

    def _setup_mediapipe(self):
        """Initializes the face mesh detection if video is enabled."""
        if not self.enable_video:
            self.get_logger().info("Video disabled. Skipping MediaPipe initialization.")
            return

        self.get_logger().info("Initializing MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _setup_recording(self):
        """Configures file writers if recording is enabled."""
        self.video_writer = None
        self.recording_fps = 20 

        if not self.record_output:
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.video_save_path = os.path.join(self.output_dir, 'video_demo.avi')
        self.audio_save_path = os.path.join(self.output_dir, 'audio_demo.wav')
        self.fused_save_path = os.path.join(self.output_dir, 'fused_demo.mp4')

        # Setup Audio Writer
        self.audio_writer = wave.open(self.audio_save_path, 'wb')
        self.audio_writer.setnchannels(1)       
        self.audio_writer.setsampwidth(2)       
        self.audio_writer.setframerate(self.audio_sample_rate)

        self.get_logger().info(f"Recording enabled. Output directory: {self.output_dir}")

    def audio_callback(self, msg):
        if not self.enable_audio:
            return
        self.audio_buffer.extend(msg.data)
        if self.record_output:
            audio_np = np.array(msg.data, dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            self.audio_writer.writeframes(audio_int16.tobytes())

    def video_callback(self, msg):
        if not self.enable_video:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_time = time.time()
            self.video_buffer.append((current_time, cv_image))
            
            if self.record_output:
                if self.video_writer is None:
                    h, w = cv_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, self.recording_fps, (w, h))

                rec_frame = cv_image.copy()
                cv2.rectangle(rec_frame, (5, 5), (450, 50), (0, 0, 0), -1)
                cv2.putText(rec_frame, self.latest_prediction_string, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.video_writer.write(rec_frame)

            # Cleanup old frames
            max_frame_age = self.window_seconds + self.video_cleanup_margin_seconds
            while self.video_buffer and (current_time - self.video_buffer[0][0]) > max_frame_age:
                self.video_buffer.popleft()
                
        except Exception as e:
            self.get_logger().error(f"Video conversion/recording error: {e}")

    def timer_callback(self):
        # 0. Check if at least one modality is active
        if not self.enable_audio and not self.enable_video:
            return

        current_time = time.time()
        
        # 1. Check Buffers dynamically based on active flags
        if self.enable_audio and len(self.audio_buffer) < self.audio_buffer_samples:
            return

        if self.enable_video:
            valid_frames = [frame for ts, frame in self.video_buffer if (current_time - ts) <= self.window_seconds]
            if len(valid_frames) < self.num_video_frames:
                return

        # ----------------------------------------------------
        # --- A. Process Video ---
        # ----------------------------------------------------
        lm_start = time.perf_counter()
        
        if self.enable_video:
            indices = np.linspace(0, len(valid_frames) - 1, self.num_video_frames, dtype=int)
            sampled_frames = [valid_frames[i] for i in indices]
            
            processed_landmarks = []
            visibility_mask = []

            for frame in sampled_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                frame_landmarks = torch.zeros(1404)
                is_visible = False

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
                    
                    nose_tip = coords[1]
                    coords = coords - nose_tip
                    
                    left_eye, right_eye = coords[33], coords[263]
                    eye_distance = np.linalg.norm(left_eye - right_eye)
                    
                    if eye_distance > 1e-6: 
                        coords = coords / eye_distance
                        
                    frame_landmarks = torch.tensor(coords.flatten(), dtype=torch.float32)
                    is_visible = True

                processed_landmarks.append(frame_landmarks)
                visibility_mask.append(is_visible)

            video_tensor = torch.stack(processed_landmarks).unsqueeze(0).to(self.device, non_blocking=True)
            
            # Use dynamic face visibility threshold
            video_valid = (sum(visibility_mask) / len(visibility_mask)) > self.face_visibility_threshold
            
            if not video_valid:
                self.get_logger().debug("WARNING: Face visibility below threshold!")
                
            video_mask = torch.tensor([video_valid], dtype=torch.bool, device=self.device)
        else:
            video_tensor = torch.zeros(1, self.num_video_frames, 1404, device=self.device)
            video_mask = torch.tensor([False], dtype=torch.bool, device=self.device)

        lm_ms = (time.perf_counter() - lm_start) * 1000.0

        # ----------------------------------------------------
        # --- B. Process Audio ---
        # ----------------------------------------------------
        tensor_start = time.perf_counter()
        
        if self.enable_audio:
            audio_np = np.array(self.audio_buffer, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device, non_blocking=True)

            rms_energy = torch.sqrt(torch.mean(audio_tensor**2)).item()
            
            # Use dynamic audio energy threshold
            if rms_energy < self.audio_energy_threshold:
                audio_mask = torch.tensor([False], dtype=torch.bool, device=self.device)
                self.get_logger().debug(f"Audio ignored. RMS ({rms_energy:.5f}) < Threshold ({self.audio_energy_threshold})")
            else:
                audio_mask = torch.tensor([True], dtype=torch.bool, device=self.device)
        else:
            audio_tensor = torch.zeros(1, self.audio_buffer_samples, device=self.device)
            audio_mask = torch.tensor([False], dtype=torch.bool, device=self.device)

        # ----------------------------------------------------
        # --- C. Inference ---
        # ----------------------------------------------------
        batch = {
            'audio': audio_tensor,  
            'video': video_tensor,  
            'audio_mask': audio_mask,
            'video_mask': video_mask
        }

        tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
        
        try:
            infer_start = time.perf_counter()
            autocast_enabled = self.device.type == 'cuda'
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = self.model(batch).squeeze(0)
            model_ms = (time.perf_counter() - infer_start) * 1000.0

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            self.ema_alpha = float(np.max(probs))

            if self.ema_probs is None:
                self.ema_probs = probs
            else:
                self.ema_probs = (self.ema_alpha * probs) + ((1.0 - self.ema_alpha) * self.ema_probs)
                
            predicted_index = int(np.argmax(self.ema_probs))
            max_prob = float(self.ema_probs[predicted_index])

            # Use dynamic confidence threshold
            if max_prob >= self.confidence_threshold:
                self.current_emotion = self.EMOTION_MAP[predicted_index]
            else:
                self.current_emotion = "Neutral"

            self.latest_prediction_string = f"[{self.current_emotion.upper()}] Conf: {max_prob:.2f}"
            
            emotion_label = (
                f"{self.current_emotion} (Conf: {max_prob:.2f}, detected emotion: {self.EMOTION_MAP[predicted_index]})"
                f" | alpha {self.ema_alpha:.2f}"
                f" | times: landmarks {lm_ms:.1f} ms, tensor {tensor_ms:.1f} ms, model {model_ms:.1f} ms"
            )
            
            output_msg = String()
            output_msg.data = emotion_label
            self.emotion_pub.publish(output_msg)
            self.get_logger().info(f"Published: {emotion_label}")

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def destroy_node(self):
        if self.record_output:
            self.get_logger().info("Halting recording... Fusing Audio and Video into MP4.")
            self.record_output = False
            
            if self.video_writer:
                self.video_writer.release()
            if hasattr(self, 'audio_writer') and self.audio_writer:
                self.audio_writer.close()
            
            try:
                subprocess.run([
                    'ffmpeg', '-y', 
                    '-i', self.video_save_path, 
                    '-i', self.audio_save_path, 
                    '-c:v', 'libx264',   
                    '-c:a', 'aac',       
                    self.fused_save_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                os.remove(self.video_save_path)
                self.get_logger().info(f"Fusion complete! Saved as '{self.fused_save_path}'")
                
            except Exception as e:
                self.get_logger().error(f"FFmpeg fusion failed: {e}")

        super().destroy_node()

class Fullmodel42(nn.Module):
    def __init__(self,encoder_weights=None,dim=256,n_classes=6,hidden_dim=128,tcn_layers=2):
        super().__init__()
        encoder = MultimodalEncoder2()
        if encoder_weights is not None:
            encoder.load_state_dict(torch.load(encoder_weights))
        self.audio_encoder = encoder.audio_encoder
        self.video_encoder = encoder.video_encoder
        self.fusion = RobustCrossAttentionFusion(dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(2*dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,n_classes)
        )
        self.missing_audio_token = nn.Parameter(torch.randn(1,dim)*0.02)
        self.missing_video_token = nn.Parameter(torch.randn(1,dim)*0.02)
    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        za = self.audio_encoder(audio)
        zv = self.video_encoder(video)
        audio_mask = batch.get('audio_mask',None)
        video_mask = batch.get('video_mask',None)
        if audio_mask is not None:
            za = torch.where(audio_mask.unsqueeze(1),za,self.missing_audio_token.expand_as(za))
        if video_mask is not None:
            zv = torch.where(video_mask.unsqueeze(1),zv,self.missing_video_token.expand_as(zv))
        z = self.fusion(za,zv)
        logits = self.classifier(z)
        return logits
    
class MultimodalEncoder2(nn.Module):
    def __init__(self,out_dim=256,tcn_layers=2,lora_layers=[4,5,6,7,8,9,10,11,12],audio_dropout=0.4):
        super().__init__()
        self.audio_encoder = WavLMAudioEncoder(out_dim,lora_layers,audio_dropout,tcn_layers)
        #self.video_encoder = DANVideoEncoder(tcn_layers=tcn_layers)
        self.video_encoder = LandmarkGRUEncoder()
        self.audio_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
    def forward(self,batch):
        audio = batch['audio']
        za = self.audio_encoder(audio)
        za = self.audio_proj(za)
        za = F.normalize(za,p=2,dim=1)
        video = batch['video']
        zv = self.video_encoder(video)
        zv = self.video_proj(zv)
        zv = F.normalize(zv,p=2,dim=1)
        return za, zv
    
class RobustCrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.attn_a2v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_v2a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1_a = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim*2)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, za, zv):
        za_seq = za.unsqueeze(1) # Diventa (Batch, 1, Dim)
        zv_seq = zv.unsqueeze(1) # Diventa (Batch, 1, Dim)
        out_a, _ = self.attn_a2v(query=za_seq, key=zv_seq, value=zv_seq)
        za_enriched = self.norm1_a(za_seq + self.dropout(out_a))
        out_v, _ = self.attn_v2a(query=zv_seq, key=za_seq, value=za_seq)
        zv_enriched = self.norm1_v(zv_seq + self.dropout(out_v))
        fused = torch.cat([za_enriched, zv_enriched], dim=-1)
        ffn_out = self.ffn(fused)
        out = self.norm2(fused + self.dropout(ffn_out))
        return out.squeeze(1)

class WavLMAudioEncoder(nn.Module):
    def __init__(self,out_dim=256,lora_layers=[4,5,6,7],dropout=0.4,n_tcn_layers=4):
        super().__init__()
        
        wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base',use_safetensors=True)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"], 
            lora_dropout=0.3,
            bias="none",
            layers_to_transform=lora_layers
        )
        self.model = get_peft_model(wavlm,config)
        '''
        self.model = WavLMModel.from_pretrained('microsoft/wavlm-base')
        for p in self.model.parameters():
            p.requires_grad_ = False
        '''
        self.pool = WeightedLayerPooling(12)
        self.proj = nn.Sequential(
            nn.Linear(768,out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tcn = LiteTCN(out_dim,n_tcn_layers)
        #self.gru = nn.GRU(input_size=out_dim, hidden_size=out_dim//2, 
                          #num_layers=1, batch_first=True, bidirectional=True)
        #self.tpool = AttentiveTemporalPooling(out_dim)
        #self.tpool = StdPooling(256)
    def forward(self,audio):
        # audio: (B,16000)
        with torch.no_grad():
            out = self.model(audio,output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states # each is (B,T,D)
        layers = [hidden_states[l] for l in list(range(13))]
        z = self.pool(layers)
        z = self.proj(z) # (B,49,256)
        z = self.tcn(z) 
        #return self.tpool(z)
        return z.mean(dim=1)
        #out = self.tpool(z)
        #out = torch.cat([h_n[0], h_n[1]], dim=1) 
        #return out

class LandmarkGRUEncoder(nn.Module):
    def __init__(self,in_features=1404,hidden_dim=128,out_dim=256,n_layers=2,dropout=0.4):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_features,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim*2,out_dim)
    def forward(self,x):
        B,T,F = x.size()
        x_flat = x.view(B*T,F)
        spatial_feats = self.spatial_mlp(x_flat)
        spatial_feats = spatial_feats.view(B,T,-1)
        _,hidden = self.gru(spatial_feats)
        zv = torch.cat([hidden[-2], hidden[-1]],dim=1)
        return self.proj(zv)

class WeightedLayerPooling(nn.Module):
    def __init__(self,n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers))
    def forward(self,input):
        w = F.softmax(self.weights,dim=0)
        out = 0.0
        for wi,li in zip(w,input):
            out = out + wi*li
        return out

class LiteTCN(nn.Module):
    def __init__(self,channels,num_layers=2):
        super().__init__()
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels,kernel_size=3,dilation=d
            ) for d in dilations
        ])
    def forward(self,x):
        # x: (B,T,C)
        x = x.transpose(1,2) # (B,C,T)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1,2) # (B,T,C)
    
class DepthwiseSeparableTCNBlock(nn.Module):
    def __init__(self,channels,kernel_size=3,dilation=1):
        super().__init__()
        padding = (kernel_size-1)*dilation
        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels
        )
        self.pointwise = nn.Conv1d(channels,channels,kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
    def forward(self,x):
        # x: (B,C,T)
        out = self.depthwise(x)
        out = out[:,:,:-self.depthwise.padding[0]] # causal trim
        out = self.pointwise(out)
        out = self.norm(out)
        return F.relu(out + x)

def main(args=None):
    rclpy.init(args=args)
    node = EmotionRecognitionNode()  
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Emotion Node.")
    finally:
        if hasattr(node, 'face_mesh'):
            node.face_mesh.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
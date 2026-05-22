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
from emotion_pkg.model import Fullmodel42, FullmodelE2E
import wave
import subprocess
import os

class EmotionRecognitionNode(Node):
    def __init__(self, model_mode=None,record=False):
        super().__init__('emotion_recognition_node')
        self.bridge = CvBridge()

        # --- 0. Runtime Mode Setup ---
        self.declare_parameter('model', 'short')
        param_model_mode = self.get_parameter('model').get_parameter_value().string_value.lower().strip()

        if model_mode is not None:
            selected_model_mode = str(model_mode).lower().strip()
        else:
            selected_model_mode = param_model_mode

        if selected_model_mode not in ('long', 'short'):
            self.get_logger().warn(
                f"Invalid model value '{selected_model_mode}'. Falling back to 'short'."
            )
            selected_model_mode = 'short'

        self.model_mode = selected_model_mode
        self.window_seconds = 3.0 if self.model_mode == 'long' else 1.0
        self.num_video_frames = 16 if self.model_mode == 'long' else 8
        self.model_weights_path = (
            '/home/franc/ros2-ws/src/emotion_pkg/emotion_pkg/final_model_3s.pt'
            if self.model_mode == 'long'
            else '/home/franc/ros2-ws/src/emotion_pkg/emotion_pkg/final_model_1s.pt'
        )
        self.audio_sample_rate = 16000
        self.audio_buffer_samples = int(self.audio_sample_rate * self.window_seconds)

        # Keep enough frame history for the selected window + small margin
        self.video_assumed_fps = 40
        self.video_buffer_maxlen = int(self.video_assumed_fps * self.window_seconds)
        self.video_cleanup_margin_seconds = 0.5

        # Prefer GPU if available for model inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        
        # --- 1. AI Model Setup ---
        self.get_logger().info("Waking up the Neural Network...")
        
        try:
            # Initialize your model's architecture
            self.model = Fullmodel42(dim=256)
            self.model.load_state_dict(torch.load(self.model_weights_path, map_location=torch.device('cpu')))
            
            # Lock the model in evaluation mode
            self.model.eval()
            self.model.to(self.device)
            device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'
            self.get_logger().info(
                f"Neural Network loaded successfully on {device_name}. "
                f"Mode: {self.model_mode}, window: {self.window_seconds:.0f}s, frames: {self.num_video_frames}"
            )
            
        except Exception as e:
            self.get_logger().error(f"FATAL: Could not load the model. Error: {e}")
            raise SystemExit

        # --- 2. MediaPipe Setup ---
        self.get_logger().info("Initializing MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- 3. Buffer Setup ---
        self.audio_buffer = deque(maxlen=self.audio_buffer_samples)
        self.video_buffer = deque(maxlen=self.video_buffer_maxlen)

        # --- 4. Subscribers & Publishers ---
        self.video_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.video_callback, 10)
        self.audio_sub = self.create_subscription(
            Float32MultiArray, '/microphone/audio_data', self.audio_callback, 10)

        self.emotion_pub = self.create_publisher(String, '/robot/detected_emotion', 10)
        
        # --- 5. Timer ---
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.EMOTION_MAP = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad'
        }
        
        # --- 6. Inference Smoothing Parameters ---
        self.ema_probs = None
        self.ema_alpha = 0.4
        self.conf_threshold = 0.5
        self.current_emotion = "Neutral"

        # --- 7. Recording Setup (New) ---
        self.record_output = record # Set to False to disable recording
        self.video_writer = None
        self.latest_prediction_string = "Initializing..."
        self.save_path = '/home/franc/ros2-ws/build/emotion_pkg/emotion_pkg/video_demo.avi' # Adjust path
        
        # Define codec and create VideoWriter (Lightweight XVID)
        # Using 15 or 20 FPS is usually enough for a presentation and saves CPU
        self.recording_fps = 20 

        self.audio_save_path = '/home/franc/ros2-ws/build/emotion_pkg/emotion_pkg/audio_demo.wav'
        self.audio_writer = wave.open(self.audio_save_path, 'wb')
        self.audio_writer.setnchannels(1)       # Mono
        self.audio_writer.setsampwidth(2)       # 16-bit audio (2 bytes)
        self.audio_writer.setframerate(16000)   # Must match your mic_node rate

        if record:
            self.get_logger().info(f"Recording session started. Saving video to {self.save_path}")
        self.fused_save_path = '/home/franc/ros2-ws/build/emotion_pkg/emotion_pkg/fused_demo.mp4'

    def audio_callback(self, msg):
        """ Appends new audio chunks to the rolling buffer. """
        self.audio_buffer.extend(msg.data)
        if self.record_output:
            # ROS 2 Float32 arrays (-1.0 to 1.0) must be converted to Int16 for standard WAV files
            audio_np = np.array(msg.data, dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            self.audio_writer.writeframes(audio_int16.tobytes())

    def video_callback(self, msg):
        """ Appends frames to buffer and writes to disk with overlay. """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_time = time.time()
            self.video_buffer.append((current_time, cv_image))
            
            # --- Recording Logic ---
            if self.record_output:
                if self.video_writer is None:
                    h, w = cv_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Lightweight codec
                    self.video_writer = cv2.VideoWriter(self.save_path, fourcc, self.recording_fps, (w, h))

                # Draw the overlay on a copy for the video file
                rec_frame = cv_image.copy()
                
                # Visual Box for the label
                cv2.rectangle(rec_frame, (5, 5), (450, 50), (0, 0, 0), -1)
                cv2.putText(rec_frame, self.latest_prediction_string, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.video_writer.write(rec_frame)

            # --- (Existing Cleanup Logic) ---
            max_frame_age = self.window_seconds + self.video_cleanup_margin_seconds
            while self.video_buffer and (current_time - self.video_buffer[0][0]) > max_frame_age:
                self.video_buffer.popleft()
                
        except Exception as e:
            self.get_logger().error(f"Video conversion/recording error: {e}")

    
    def destroy_node(self):
        if self.record_output:
            self.get_logger().info("Halting recording... Fusing Audio and Video into MP4.")
            self.record_output = False
            
            # 1. Close the files safely
            self.video_writer.release()
            self.audio_writer.close()
            
            # 2. Command FFmpeg to fuse them (This takes about 1-2 seconds)
            try:
                subprocess.run([
                    'ffmpeg', '-y', 
                    '-i', self.save_path, 
                    '-i', self.audio_save_path, 
                    '-c:v', 'libx264',   # Compress video to standard MP4
                    '-c:a', 'aac',       # Compress audio to standard AAC
                    self.fused_save_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 3. Clean up the messy temp files
                os.remove(self.save_path)
                #os.remove(self.audio_save_path)
                self.get_logger().info("Fusion complete! Saved as 'synced_output.mp4'")
                
            except Exception as e:
                self.get_logger().error(f"FFmpeg fusion failed: {e}")

        # Call the standard ROS 2 shutdown procedures
        super().destroy_node()

    def timer_callback(self):
        current_time = time.time()
        
        # 1. Check Buffers
        if len(self.audio_buffer) < self.audio_buffer_samples:
            return

        valid_frames = [frame for ts, frame in self.video_buffer if (current_time - ts) <= self.window_seconds]
        if len(valid_frames) < self.num_video_frames:
            return

        # 2. Sample frames uniformly (16 for long mode, 8 for short mode)
        indices = np.linspace(0, len(valid_frames) - 1, self.num_video_frames, dtype=int)
        sampled_frames = [valid_frames[i] for i in indices]
        
        # --- PyTorch Tensor Formatting ---

        # Timing: measure each stage in ms
        lm_start = time.perf_counter()

        # A. Process Video: Extract landmarks for sampled frames
        processed_landmarks = []
        visibility_mask = []

        for frame in sampled_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            frame_landmarks = torch.zeros(1404)
            is_visible = False

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                
                # ==========================================================
                # --- NEW: INTER-PUPILLARY NORMALIZATION (NumPy) ---
                # ==========================================================
                # Convert MediaPipe landmarks to a (468, 3) numpy array
                coords = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
                
                # 1. Center the face on the nose tip (MediaPipe index 1)
                nose_tip = coords[1]
                coords = coords - nose_tip
                
                # 2. Calculate Euclidean distance between outer eye corners
                left_eye = coords[33]
                right_eye = coords[263]
                eye_distance = np.linalg.norm(left_eye - right_eye)
                
                # 3. Scale the entire mesh by the eye distance
                if eye_distance > 1e-6: # Prevent division by zero
                    coords = coords / eye_distance
                    
                # Flatten to a 1D tensor of 1404 elements for the model
                frame_landmarks = torch.tensor(coords.flatten(), dtype=torch.float32)
                is_visible = True
                # ==========================================================

            processed_landmarks.append(frame_landmarks)
            visibility_mask.append(is_visible)

        lm_ms = (time.perf_counter() - lm_start) * 1000.0

        # Stack into (T, 1404) and add Batch dimension -> [1, T, 1404]
        tensor_start = time.perf_counter()
        video_tensor = torch.stack(processed_landmarks).unsqueeze(0).to(self.device, non_blocking=True)

        # Calculate dynamic video mask (True if face is visible for > 50% of the frames)
        vis_ratio = sum(visibility_mask) / len(visibility_mask)
        video_valid = vis_ratio > 0.5
        if video_valid == False:
            self.get_logger().debug(f"WARNING: Face was not detected!")
        video_mask = torch.tensor([video_valid], dtype=torch.bool)

        # B. Process Audio: (48000) -> [1, 48000]
        audio_np = np.array(self.audio_buffer, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device, non_blocking=True)

        # RMS check
        rms_energy = torch.sqrt(torch.mean(audio_tensor**2)).item()
        #self.get_logger().info(f"RMS Energy: {rms_energy:.5f}")
        energy_threshold = 0.01 
        if rms_energy < energy_threshold:
            # The room is quiet; force the model to rely only on video
            audio_mask = torch.tensor([False], dtype=torch.bool, device=self.device)
            
            # Optional: Log this so you know when the robot is ignoring the mic
            self.get_logger().debug(f"Audio ignored. RMS ({rms_energy:.5f}) < Threshold")
        else:
            # Someone is speaking loudly enough
            audio_mask = torch.tensor([True], dtype=torch.bool, device=self.device)
        
        # Audio is always considered valid if the buffer was full
        #audio_mask = torch.tensor([False], dtype=torch.bool) 

        # C. Create the Batch Dictionary
        batch = {
            'audio': audio_tensor,  
            'video': video_tensor,  
            'audio_mask': audio_mask,
            'video_mask': video_mask.to(self.device)
        }

        tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
        
        # --- Inference ---
        try:
            infer_start = time.perf_counter()
            autocast_enabled = self.device.type == 'cuda'
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    logits = self.model(batch).squeeze(0)
            model_ms = (time.perf_counter() - infer_start) * 1000.0

            # 1. Convert raw logits to probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # 2. Confidence-based EMA alpha
            self.ema_alpha = float(np.max(probs))

            # 3. Apply EMA smoothing
            if self.ema_probs is None:
                self.ema_probs = probs
            else:
                self.ema_probs = (self.ema_alpha * probs) + ((1.0 - self.ema_alpha) * self.ema_probs)
                
            # 4. Confidence threshold decision logic
            predicted_index = int(np.argmax(self.ema_probs))
            max_prob = float(self.ema_probs[predicted_index])

            if max_prob >= self.conf_threshold:
                self.current_emotion = self.EMOTION_MAP[predicted_index]
            else:
                self.current_emotion = "Neutral"

            self.latest_prediction_string = f"[{self.current_emotion.upper()}] Conf: {max_prob:.2f}"
            
            # Format output for easy debugging
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

def main(args=None, model=None):
    rclpy.init(args=args)
    node = EmotionRecognitionNode(model_mode='short',record = True)  # You can set 'long' or 'short' here or via ROS2 parameter
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Emotion Node.")
    finally:
        node.face_mesh.close() # Cleanly close MediaPipe
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
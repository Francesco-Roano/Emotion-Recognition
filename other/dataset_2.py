import os
import torch
import random
from torch.utils.data import Dataset
import librosa
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
import torchaudio
from torchvision.transforms import v2

class CremaDSmartLoader(Dataset):
    def __init__(self,root_dir="/home/roano/standalone/crema-d",duration=1.0,n_frames=16,resolution=224,augmentor=None,modality_dropout_prob=0.0):
        self.dataset_root = root_dir
        self.audio_dir = os.path.join(self.dataset_root, "AudioWAV")
        #self.video_dir = os.path.join(self.dataset_root, "VideoMp4")
        self.video_dir = os.path.join(self.dataset_root, "VideoCropped")
        self.duration = duration
        self.n_frames = n_frames
        self.resolution = resolution
        self.modality_dropout_prob = modality_dropout_prob
        csv_path = os.path.join(self.dataset_root,"finishedResponses.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Can't locate CSV file in: {csv_path}")
        df = pd.read_csv(csv_path,low_memory=False)
        self.df = df.groupby("clipName")["respEmo"].agg(lambda x: x.mode()[0]).reset_index() 
        self.emotion2idx = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'S': 5}
        self.augmentor = augmentor
        self.mode = 'train'

    def __len__(self):
        return len(self.df)
    
    def get_start_idx(self,waveform,sr):
        n_samples = waveform.shape[1]
        win_size = self.max_audio_len
        if n_samples <= win_size:
            return 0, n_samples, 0.0
        stride = int(0.1 * sr)
        windows = waveform.unfold(1,win_size,stride)
        energies = torch.sum(windows**2, dim=(0,2))
        max_idx = torch.argmax(energies).item()
        start_sample = max_idx * stride
        end_sample = start_sample + win_size
        start_sec = start_sample / sr
        return start_sample, end_sample, start_sec
    
    def __getitem__(self, idx):
        
        # load label
        row = self.df.iloc[idx]
        clip = row['clipName']
        emo = row["respEmo"]
        label = self.emotion2idx[emo]

        # audio/video loading & synchronization
        audio_path = os.path.join(self.audio_dir, clip + '.wav')
        audio, sr = librosa.load(audio_path,sr=None)
        self.max_audio_len = int(self.duration * sr)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        self.sr = sr
        total_audio_len = len(audio)

        video_path = os.path.join(self.video_dir, clip + '.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open {video_path}")
        fps= cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # use window with highest energy
        start_sample, end_sample, start_sec = self.get_start_idx(waveform,sr)

        # audio crop
        audio_crop = audio[start_sample:end_sample]
        if len(audio_crop)<self.max_audio_len:
            audio_crop = np.pad(audio_crop, (0,self.max_audio_len-len(audio_crop)))
        audio_tensor = torch.tensor(audio_crop,dtype=torch.float32)
        if self.augmentor is not None:
            if self.mode == 'train':
                audio_tensor = self.augmentor.augment_audio(audio_tensor)

        # video crop
        total_frames = int(self.fps * self.duration)
        start_frame = int(start_sec*fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            frames = [np.zeros((self.resolution, self.resolution, 3),dtype=np.uint8)] * self.n_frames
        elif len(frames) < self.n_frames:
            frames += [frames[-1]]*(self.n_frames - len(frames))
        elif len(frames) > self.n_frames:
            indices = np.linspace(0, len(frames)-1, self.n_frames).astype(int)
            frames = [frames[i] for i in indices]
        processed = []
        for f in frames:
            f = cv2.resize(f, (self.resolution,self.resolution))
            f = torch.tensor(f,dtype=torch.float32).permute(2,0,1)/255.0
            processed.append(f)
        video_tensor = torch.stack(processed)
        video_tensor = v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])(video_tensor)
        if self.augmentor is not None:
            if self.mode == 'train':
                video_tensor = self.augmentor.train_video_transform(video_tensor)
            #elif self.mode == 'eval':
                #video_tensor = self.augmentor.eval_video_transform(video_tensor)

        # Modality Dropout (only in train mode)
        audio_mask = True
        video_mask = True
        
        if self.mode == 'train' and self.modality_dropout_prob > 0.0:
            p = random.random()
            # Drop audio or video, but never both
            if p < self.modality_dropout_prob:
                #audio_tensor = torch.zeros_like(audio_tensor)
                audio_mask = False
            elif p < 2 * self.modality_dropout_prob:
                #video_tensor = torch.zeros_like(video_tensor)
                video_mask = False
        

        # extract speaker
        speaker = int(clip[:4])

        # return dictionary with everything
        return {
            'audio':audio_tensor, 
            'video':video_tensor, 
            'label':torch.tensor(label, dtype=torch.long), 
            'speaker':torch.tensor(speaker, dtype=torch.long),
            'audio_mask': torch.tensor(audio_mask, dtype=torch.bool),
            'video_mask': torch.tensor(video_mask, dtype=torch.bool)
        }
        


class CremaDLoader(Dataset):
    def __init__(self,root_dir="/home/roano/standalone/crema-d",max_audio_len=16000,n_frames=16,resolution=224):
        self.dataset_root = root_dir
        self.audio_dir = os.path.join(self.dataset_root, "AudioWAV")
        self.video_dir = os.path.join(self.dataset_root, "VideoMp4")
        self.max_audio_len = max_audio_len
        self.n_frames = n_frames
        self.resolution = resolution
        csv_path = os.path.join(self.dataset_root,"finishedResponses.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Can't locate CSV file in: {csv_path}")
        df = pd.read_csv(csv_path,low_memory=False)
        self.df = df.groupby("clipName")["respEmo"].agg(lambda x: x.mode()[0]).reset_index() 
        self.emotion2idx = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'S': 5}

    def __len__(self):
        return len(self.df)
    
    def parameters(self):
        return self.sr,self.fps
    
    def __getitem__(self, idx):
        
        # load label
        row = self.df.iloc[idx]
        clip = row['clipName']
        emo = row["respEmo"]
        label = self.emotion2idx[emo]

        # audio/video loading & synchronization
        audio_path = os.path.join(self.audio_dir, clip + '.wav')
        audio, sr = librosa.load(audio_path,sr=None)
        self.sr = sr
        total_audio_len = len(audio)
        total_audio_sec = total_audio_len/sr

        video_path = os.path.join(self.video_dir, clip + '.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open {video_path}")
        fps= cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_sec = total_frames/fps

        total_sec = min(total_audio_sec,total_video_sec)
        crop_sec = self.max_audio_len/sr
        if total_sec <= crop_sec:
            start_sec = 0.0
        else:
            start_sec = random.uniform(0.0,total_sec-crop_sec)

        # audio crop
        start_sample = int(start_sec*sr)
        end_sample = start_sample + self.max_audio_len
        audio_crop = audio[start_sample:end_sample]
        if len(audio_crop)<self.max_audio_len:
            audio_crop = np.pad(audio_crop, (0,self.max_audio_len-len(audio_crop)))
        audio_tensor = torch.tensor(audio_crop,dtype=torch.float32)
        rms = torch.sqrt(torch.mean(audio_tensor**2))
        target_rms = 0.03
        audio_tensor = audio_tensor / (rms + 1e-7) * target_rms

        # video crop
        start_frame = int(start_sec*fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        for i in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            frames = [np.zeros((self.resolution, self.resolution, 3),dtype=np.uint8)] * self.n_frames
        elif len(frames) < self.n_frames:
            frames += [frames[-1]]*(self.n_frames - len(frames))
        processed = []
        for f in frames:
            f = cv2.resize(f, (self.resolution,self.resolution))
            f = torch.tensor(f,dtype=torch.float32).permute(2,0,1)/255.0
            processed.append(f)
        video_tensor = torch.stack(processed)

        # return dictionary with everything
        return {'audio':audio_tensor, 'video':video_tensor, 'label':torch.tensor(label, dtype=torch.long)}
    
class CremaDLoader2(Dataset):
    def __init__(self,
                 max_audio_len=16000,
                 n_frames=16,
                 resolution=224):
        
        self.dataset_root = "C:/Users/franc/crema-d-mirror/data/crema-d"
        self.audio_dir = os.path.join(self.dataset_root, "AudioWAV")
        self.video_dir = os.path.join(self.dataset_root, "VideoMp4")
        self.max_audio_len = max_audio_len
        self.n_frames = n_frames
        self.resolution = resolution
        self.audio_files = sorted([
            f for f in os.listdir(self.audio_dir)
            if f.lower().endswith(".wav")
        ])
        if len(self.audio_files) == 0:
            raise ValueError("No .wav file foung!")
        self.emotion_map = {
            "ANG": 0,
            "DIS": 1,
            "FEA": 2,
            "HAP": 3,
            "NEU": 4,
            "SAD": 5,
        }

    def extract_label(self,filename):
        fname = os.path.basename(filename).replace(".wav", "")
        parts = fname.split("_")
        emo_code = parts[2]
        return self.emotion_map[emo_code] 

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_name = self.audio_files[idx]
        video_name = audio_name.replace(".wav", ".mp4")
        
        # load label
        label = self.extract_label(audio_name)
        
        # audio/video loading & synchronization
        audio_path = os.path.join(self.audio_dir, audio_name)
        audio, sr = librosa.load(audio_path,sr=None)
        self.sr = sr
        total_audio_len = len(audio)
        total_audio_sec = total_audio_len/sr

        video_path = os.path.join(self.video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open {video_path}")
        fps= cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_sec = total_frames/fps

        total_sec = min(total_audio_sec,total_video_sec)
        crop_sec = self.max_audio_len/sr
        if total_sec <= crop_sec:
            start_sec = 0.0
        else:
            start_sec = random.uniform(0.0,total_sec-crop_sec)

        # audio crop
        start_sample = int(start_sec*sr)
        end_sample = start_sample + self.max_audio_len
        audio_crop = audio[start_sample:end_sample]
        if len(audio_crop)<self.max_audio_len:
            audio_crop = np.pad(audio_crop, (0,self.max_audio_len-len(audio_crop)))
        audio_tensor = torch.tensor(audio_crop,dtype=torch.float32)
        rms = torch.sqrt(torch.mean(audio_tensor**2))
        target_rms = 0.03
        audio_tensor = audio_tensor / (rms + 1e-7) * target_rms


        # video crop
        start_frame = int(start_sec*fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        for i in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            frames = [np.zeros((self.resolution, self.resolution, 3),dtype=np.uint8)] * self.n_frames
        elif len(frames) < self.n_frames:
            frames += [frames[-1]]*(self.n_frames - len(frames))
        processed = []
        for f in frames:
            f = cv2.resize(f, (self.resolution,self.resolution))
            f = torch.tensor(f,dtype=torch.float32).permute(2,0,1)/255.0
            processed.append(f)
        video_tensor = torch.stack(processed)

        # return dictionary with everything
        return {'audio':audio_tensor, 'video':video_tensor, 'label':torch.tensor(label, dtype=torch.long)}

class CremaEncodedDataset(Dataset):
    def __init__(self, smart=False):
        """
        Args:
            folder_path (string): Percorso alla cartella contenente i file .pt
        """
        if smart:
            self.folder_path = '/home/roano/standalone/crema-d-encoded-smart'
        else:
            self.folder_path = '/home/roano/standalone/crema-d-encoded'
        # Crea una lista di tutti i file .pt nella cartella
        self.files = [f for f in os.listdir(self.folder_path) if f.endswith('.pt')]
        
        # Check di sicurezza
        if len(self.files) == 0:
            print(f"Attenzione: Nessun file .pt trovato in {self.folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Recupera il nome del file
        filename = self.files[idx]
        file_path = os.path.join(self.folder_path, filename)
        
        # 2. Carica il tensore concatenato (Audio + Video)
        # map_location='cpu' è importante per non saturare la GPU durante il caricamento
        full_tensor = torch.load(file_path, map_location='cpu').detach()
        
        # 3. Estrazione Label dal nome file
        # Formato atteso: "NomeClip_Label.pt" -> es. "1001_IEO_3.pt"
        # splitext toglie l'estensione (.pt), [-1] prende l'ultimo carattere
        try:
            filename_no_ext = os.path.splitext(filename)[0]
            label_char = filename_no_ext[-1] # Prende l'ultimo carattere
            label = int(label_char)
            speaker = int(filename_no_ext[:4])
        except ValueError:
            # Fallback di sicurezza se il nome file non è nel formato giusto
            print(f"Errore parsing label per file: {filename}")
            label = 0 

        # 4. Slicing del Tensore
        # Audio: prime 49 colonne
        audio_features = full_tensor[:49]
        
        # Video: ultime 3656 colonne
        # Usiamo -3656: per prendere esattamente le ultime, indipendentemente dall'audio
        video_features = full_tensor[-3136:]

        # Convertiamo la label in tensore Long (formato standard per classificazione)
        label_tensor = torch.tensor(label, dtype=torch.long)

        speaker_tensor = torch.tensor(speaker, dtype=torch.long)

        return {'audio':audio_features,'video':video_features,'label':label_tensor,'speaker':speaker_tensor}
    
class DataAugmentor(nn.Module):
    def __init__(self,sr=16000):
        super().__init__()
        # Video - added stronger augmentations for "in-the-wild" AffectNet style
        self.train_video_transform = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            #v2.RandomErasing(p=0.05, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
        self.eval_video_transform = v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def augment_audio(self,waveform):
        # Audio augmentations
        if torch.rand(1)<0.5:
            gain = torch.rand(1) * 0.4 + 0.8
            waveform = waveform * gain
        if torch.rand(1) < 0.5:
            noise_level = torch.rand(1) * 0.05
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        # Simple SpecAugment-style time masking in raw waveform (zeroing out random segments)
        if torch.rand(1) < 0.3:
            mask_len = int(waveform.shape[-1] * 0.1) # mask up to 10%
            if mask_len > 0:
                start = torch.randint(0, waveform.shape[-1] - mask_len, (1,)).item()
                waveform[start:start+mask_len] = 0.0
                
        return waveform
        

class ProcessedCremaDLoader(Dataset):
    def __init__(self, data_dir='/home/roano/standalone/data/cremad_smart_processed', mode='train', augmentor=None, modality_dropout_prob=0.0,normalize=False):
        """
        Args:
            data_dir: Path to the folder containing the processed .pt files.
            mode: 'train' or 'eval'.
            augmentor: Your existing DataAugmentor class (handles audio augmentations).
            modality_dropout_prob: Probability of artificially dropping a modality during training.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.augmentor = augmentor
        self.modality_dropout_prob = modality_dropout_prob
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        
        # Load the dictionary saved by our extraction script
        data = torch.load(file_path, map_location='cpu')
        
        audio = data['audio']  # Shape: (1, audio_len)
        video = data['video']  # Shape: (16, 1404)
        mask = data['mask']    # Shape: (16,) boolean
        label = data['label']  # Scalar

        if self.normalize:
            n_frames = video.shape[0]
            video_coords = video.view(n_frames, 468, 3).clone()
        
            for i in range(n_frames):
                # 1. Center the face on the nose tip (MediaPipe index 1)
                nose_tip = video_coords[i, 1, :].clone()
                video_coords[i] = video_coords[i] - nose_tip
                
                # 2. Calculate Euclidean distance between outer eye corners
                # Left eye outer corner: index 33, Right eye outer corner: index 263
                left_eye = video_coords[i, 33, :]
                right_eye = video_coords[i, 263, :]
                eye_distance = torch.norm(left_eye - right_eye)
                
                # 3. Scale the entire mesh by the eye distance
                if eye_distance > 1e-6: # Prevent division by zero
                    video_coords[i] = video_coords[i] / eye_distance
                    
            # Flatten the tensor back to (16, 1404) for the GRU Encoder
            video = video_coords.view(n_frames, 1404)
        
        # Re-extract speaker from filename (e.g., 1001_DFA_ANG_XX.pt)
        try:
            speaker = int(self.files[idx][:4])
        except ValueError:
            speaker = 0

        # --- AUDIO & VIDEO AUGMENTATION ---
        if self.mode == 'train' and self.augmentor is not None:
            audio = self.augmentor.augment_audio(audio)
            video = self.augmentor.augment_video(video)

        # --- VISIBILITY MASK LOGIC ---
        # MediaPipe gave us a boolean for every frame. 
        # If the face was visible for MORE than 50% of the frames, we consider the video valid.
        visibility_ratio = mask.float().mean()
        video_mask = visibility_ratio > 0.5
        
        audio_mask = True
        
        # --- MODALITY DROPOUT (Training only) ---
        if self.mode == 'train' and self.modality_dropout_prob > 0.0:
            p = random.random()
            # Drop audio or video, but never both
            if p < self.modality_dropout_prob:
                audio_mask = False
            elif p < 2 * self.modality_dropout_prob:
                video_mask = False

        return {
            'audio': audio,
            'video': video,
            'label': label,
            'speaker': torch.tensor(speaker, dtype=torch.long),
            'audio_mask': torch.tensor(bool(audio_mask), dtype=torch.bool),
            'video_mask': torch.tensor(bool(video_mask), dtype=torch.bool)
        }


class MultimodalAugmentor:
    """
    Handles robust augmentations for Audio (wav) and Video (468 landmarks).
    Apply transforms with specific probabilities to ensure non-repetitive batches.
    """
    def __init__(self, p_audio=0.5, p_video=0.8, sample_rate=16000):
        self.p_audio = p_audio
        self.p_video = p_video
        self.sample_rate = sample_rate

    # --- AUDIO TRANSFORMATIONS ---
    def augment_audio(self, audio):
        # audio shape: (1, samples)
        if random.random() < self.p_audio:
            # 1. Pitch Shift (gentle shift to vary speaker voice)
            if random.random() < 0.3:
                n_steps = random.randint(-2, 2)
                audio = torchaudio.functional.pitch_shift(audio, self.sample_rate, n_steps)
            
            # 2. Gaussian Noise (simulates sensor noise)
            if random.random() < 0.3:
                noise = torch.randn_like(audio) * 0.008 # Mild noise level
                audio = audio + noise
                
            # 3. Time Masking (forces network to ignore parts of speech)
            if random.random() < 0.4:
                T = 50 # Max mask length (samples)
                mask_idx = random.randint(0, audio.size(1) - T)
                audio[:, mask_idx : mask_idx + T] = 0
                
        # (Maintain original mono/stereo format)
        return audio

    # --- VIDEO LANDMARK TRANSFORMATIONS (Coordinate Geometry) ---
    def augment_video(self, video):
        # video shape: (16, 1404), where 1404 is (468 landmarks * 3 coordinates)
        if random.random() < self.p_video:
            T, F = video.size()
            
            # Flatten only to make 3D coordinates explicit: (T, 468, 3)
            video_coords = video.view(T, 468, 3) 
            
            # 1. Scaling (Simulates distance from camera)
            if random.random() < 0.4:
                scale = random.uniform(0.85, 1.15)
                # Scale from the center of the face, not the origin (0,0,0)
                center = video_coords.mean(dim=(0, 1), keepdim=True)
                video_coords = (video_coords - center) * scale + center
                
            # 2. Rotation (Mild shift to vary pose)
            if random.random() < 0.4:
                # Max rotation angle in radians (e.g., +/- 14 degrees)
                angle_rad = random.uniform(-0.25, 0.25)
                cos_a, sin_a = torch.cos(torch.tensor(angle_rad)), torch.sin(torch.tensor(angle_rad))
                # 2D rotation matrix for (x, y) coordinates
                rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
                
                # Rotate X and Y, leave Z alone
                # (16, 468, 2) * (2, 2)
                video_coords[:, :, :2] = torch.matmul(video_coords[:, :, :2], rot_matrix)
                
            # 3. Micro-Jitter (Local coordinate noise - VERY EFFECTIVE)
            # HIGH probability because it changes precise coordinates, preventing memorization
            if random.random() < 0.8:
                # Add microscopic gaussian noise to every (x,y,z) coordinate independently
                noise = torch.randn_like(video_coords) * 0.01
                video_coords = video_coords + noise
                
            # 4. Translation (Simulates face centering variation)
            if random.random() < 0.5:
                # Shift all (x,y,z) coordinates together by a random value
                shift = (torch.rand(3) - 0.5) * 0.12
                video_coords = video_coords + shift.view(1, 1, 3)
                
            # Reshape back to required (16, 1404) flat format
            video = video_coords.view(T, F)
            
        return video
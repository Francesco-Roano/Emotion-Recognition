import os
import torch
import random
from torch.utils.data import Dataset
import librosa
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2

class CremaDSmartLoader(Dataset):
    def __init__(self,root_dir="/home/roano/standalone/crema-d",duration=1.0,n_frames=16,resolution=224,augmentor=None):
        self.dataset_root = root_dir
        self.audio_dir = os.path.join(self.dataset_root, "AudioWAV")
        self.video_dir = os.path.join(self.dataset_root, "VideoMp4")
        self.duration = duration
        self.n_frames = n_frames
        self.resolution = resolution
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
        if self.augmentor is not None:
            if self.mode == 'train':
                video_tensor = self.augmentor.train_video_transform(video_tensor)
            elif self.mode == 'eval':
                video_tensor = self.augmentor.eval_video_transform(video_tensor)

        # extract speaker
        speaker = int(clip[:4])

        # return dictionary with everything
        return {'audio':audio_tensor, 'video':video_tensor, 'label':torch.tensor(label, dtype=torch.long), 'speaker':torch.tensor(speaker, dtype=torch.long)}
        


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
        # Video
        self.train_video_transform = v2.Compose([
            v2.RandomResizedCrop(size=(224,224),scale=(0.8,1.0),antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.eval_video_transform = v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def augment_audio(self,waveform):
        if torch.rand(1)<0.5:
            gain = torch.rand(1) * 0.4 + 0.8
            waveform = waveform * gain
        if torch.rand(1) < 0.5:
            noise_level = torch.rand(1) * 0.02
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        return waveform
        



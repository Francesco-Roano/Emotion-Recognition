import os
import cv2
import torch
import torchaudio
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

class CremaDPreprocessor:
    def __init__(self, input_dir, output_dir, dataset_root, duration=3.0, num_frames=16, target_sr=16000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset_root = dataset_root
        self.duration = duration
        self.num_frames = num_frames
        self.target_sr = target_sr
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- NEW LABEL LOGIC (Human Responses Mode) ---
        csv_path = os.path.join(self.dataset_root, "finishedResponses.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Can't locate CSV file in: {csv_path}")
            
        df = pd.read_csv(csv_path, low_memory=False)
        self.df = df.groupby("clipName")["respEmo"].agg(lambda x: x.mode()[0]).reset_index() 
        self.emotion2idx = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'S': 5}
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_highest_energy_window(self, waveform, sr):
        """Finds the start sample of the highest energy window of `self.duration` seconds."""
        window_samples = int(self.duration * sr)
        
        if waveform.shape[1] <= window_samples:
            padded_wav = torch.zeros(1, window_samples)
            padded_wav[0, :waveform.shape[1]] = waveform[0]
            return padded_wav, 0.0
            
        sq_energy = waveform[0] ** 2
        cumsum_energy = torch.cumsum(sq_energy, dim=0)
        window_energy = cumsum_energy[window_samples:] - cumsum_energy[:-window_samples]
        
        start_sample = torch.argmax(window_energy).item()
        start_time_sec = start_sample / sr
        
        cropped_wav = waveform[:, start_sample : start_sample + window_samples]
        return cropped_wav, start_time_sec

    def process_video_window(self, video_path, start_time_sec):
        """Extracts 16 uniform frames from the target time window and runs MediaPipe."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time_sec * fps)
        end_frame = int((start_time_sec + self.duration) * fps)
        end_frame = min(end_frame, total_video_frames - 1)
        
        frame_indices = torch.linspace(start_frame, end_frame, self.num_frames).long().tolist()
        
        landmarks_sequence = []
        visibility_mask = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            frame_landmarks = torch.zeros(1404)
            is_visible = False
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    coords = []
                    for landmark in face.landmark:
                        coords.extend([landmark.x, landmark.y, landmark.z])
                        
                    frame_landmarks = torch.tensor(coords, dtype=torch.float32)
                    is_visible = True
            
            landmarks_sequence.append(frame_landmarks)
            visibility_mask.append(is_visible)
            
        cap.release()
        return torch.stack(landmarks_sequence), torch.tensor(visibility_mask, dtype=torch.bool)

    def extract_label(self, filename):
        """Extracts the label using the human response mode from the CSV."""
        clip_name = os.path.splitext(filename)[0]
        
        # Look up the clipName in the grouped dataframe
        row = self.df[self.df['clipName'] == clip_name]
        
        if not row.empty:
            emo = row.iloc[0]['respEmo']
            if emo in self.emotion2idx:
                return self.emotion2idx[emo]
                
        print(f"Warning: Clip {clip_name} not found in CSV or invalid emotion.")
        return None

    def process_dataset(self):
        video_files = [f for f in os.listdir(self.input_dir) if f.endswith('.flv') or f.endswith('.mp4')]
        print(f"Starting extraction for {len(video_files)} files...")
        
        valid_count = 0
        
        for filename in tqdm(video_files):
            file_path = os.path.join(self.input_dir, filename)
            
            # 1. Label Extraction
            label = self.extract_label(filename)
            if label is None:
                continue
                
            # 2. Audio Processing
            waveform, sr = torchaudio.load(file_path)
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            cropped_audio, start_time_sec = self.get_highest_energy_window(waveform, self.target_sr)
            
            # 3. Video Processing (Landmarks)
            landmarks, mask = self.process_video_window(file_path, start_time_sec)
            
            # 4. Save
            out_dict = {
                'audio': cropped_audio,
                'video': landmarks,      
                'mask': mask,            
                'label': torch.tensor(label, dtype=torch.long)
            }
            
            out_filename = os.path.splitext(filename)[0] + '.pt'
            torch.save(out_dict, os.path.join(self.output_dir, out_filename))
            valid_count += 1
            
        self.face_mesh.close()
        print(f"Extraction complete! Saved {valid_count} processed files to {self.output_dir}")

if __name__ == "__main__":
    # Update these paths
    INPUT_DIR = "/home/roano/standalone/crema-d/VideoMp4" # or VideoCropped depending on your setup
    OUTPUT_DIR = "/home/roano/standalone/data/cremad_smart_processed_1s"
    DATASET_ROOT = "/home/roano/standalone/crema-d" # Where the CSV lives
    
    preprocessor = CremaDPreprocessor(
        input_dir=INPUT_DIR, 
        output_dir=OUTPUT_DIR,
        dataset_root=DATASET_ROOT,
        duration=1.0,
        num_frames=8
    )
    preprocessor.process_dataset()
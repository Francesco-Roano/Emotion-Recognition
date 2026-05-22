from model2 import WavLMEmbedding, TimeSformerEmbedding
import os
import pandas as pd
import librosa
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm
from dataset import CremaDSmartLoader
from torch.utils.data import DataLoader

def main():

    root = "/home/roano/standalone/crema-d"
    out_folder = '/home/roano/standalone/crema-d-encoded'
    max_audio_len = 16000
    resolution = 224
    n_frames = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ",device)

    csv_path = os.path.join(root,"finishedResponses.csv")
    df = pd.read_csv(csv_path,low_memory=False)
    df = df.groupby("clipName")["respEmo"].agg(lambda x: x.mode()[0]).reset_index()
    emotion2idx = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'S': 5}
    audio_dir = os.path.join(root,'AudioWAV')
    video_dir = os.path.join(root,'VideoMp4')

    # setup embedding models
    ae = WavLMEmbedding().to(device)
    ve = TimeSformerEmbedding().to(device)

    for idx in tqdm(range(len(df))):

        #print(f"Processing {idx+1}-th /{len(df)} clip")
        
        row = df.iloc[idx]
        clip = row['clipName']
        emo = row['respEmo']
        label = emotion2idx[emo]

        # audio/video loading & synchronization
        audio_path = os.path.join(audio_dir, clip + '.wav')
        audio, sr = librosa.load(audio_path,sr=None)
        total_audio_len = len(audio)
        total_audio_sec = total_audio_len/sr

        video_path = os.path.join(video_dir, clip + '.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open {video_path}")
        fps= cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_sec = total_frames/fps

        total_sec = min(total_audio_sec,total_video_sec)
        crop_sec = max_audio_len/sr
        if total_sec <= crop_sec:
            start_sec = 0.0
        else:
            start_sec = random.uniform(0.0,total_sec-crop_sec)

        # audio crop
        start_sample = int(start_sec*sr)
        end_sample = start_sample + max_audio_len
        audio_crop = audio[start_sample:end_sample]
        if len(audio_crop)<max_audio_len:
            audio_crop = np.pad(audio_crop, (0,max_audio_len-len(audio_crop)))
        audio_tensor = torch.tensor(audio_crop,dtype=torch.float32)
        rms = torch.sqrt(torch.mean(audio_tensor**2))
        target_rms = 0.03
        audio_tensor = audio_tensor / (rms + 1e-7) * target_rms

        # audio embedding
        audio_tensor = audio_tensor.unsqueeze(0).to(device)
        za  = ae(audio_tensor)

        # video crop
        start_frame = int(start_sec*fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            frames = [np.zeros((resolution, resolution, 3),dtype=np.uint8)] * n_frames
        elif len(frames) < n_frames:
            frames += [frames[-1]]*(n_frames - len(frames))
        processed = []
        for f in frames:
            f = cv2.resize(f, (resolution,resolution))
            f = torch.tensor(f,dtype=torch.float32).permute(2,0,1)/255.0
            processed.append(f)
        video_tensor = torch.stack(processed)

        # video encoding
        video_tensor = video_tensor.unsqueeze(0).to(device)
        zv,_ = ve(video_tensor)

        # concatenation
        za = za.squeeze(0)
        zv = zv.squeeze(0).squeeze(0)
        z = torch.cat([za,zv],dim=0).cpu()

        # saving
        save_path = os.path.join(out_folder,clip+"_"+str(label)+".pt")
        try:
            torch.save(z, save_path)
        except Exception as e:
            print("[DEBUG] ERROR during saving:", e)

        
def main_2():
    out_folder = '/home/roano/standalone/crema-d-encoded-smart'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CremaDSmartLoader(n_frames=16)
    dl = DataLoader(dataset,1,shuffle=False,num_workers=4)
    ae = WavLMEmbedding().to(device)
    ve = TimeSformerEmbedding().to(device)
    for i,batch in enumerate(tqdm(dl)):
        speaker = batch['speaker'].squeeze(0)
        label = batch['label'].squeeze(0)
        save_path = os.path.join(out_folder,str(speaker.item())+'_'+str(i)+'_'+str(label.item())+'.pt')
        audio_encoded = ae(batch['audio'].to(device)).squeeze(0)
        video_encoded,_ = ve(batch['video'].to(device))
        video_encoded = video_encoded.squeeze(0).squeeze(0)
        z = torch.cat([audio_encoded,video_encoded],dim=0).cpu()
        try:
            torch.save(z, save_path)
        except Exception as e:
            print("[DEBUG] ERROR during saving:", e)
    print('DONE!')

if __name__=="__main__":
    main_2()


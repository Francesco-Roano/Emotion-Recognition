import torch
import torch.nn as nn
import torch.nn.functional as F
from model4 import WavLMAudioEncoder, DANVideoEncoder2, RobustCrossAttentionFusion
import os
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_2 import CremaDSmartLoader
from dataset import DataAugmentor
from utils import make_weighted_loaders, softF1Loss
from mercl import FocalLoss
from test_2 import test_fullmodel, save_confusion_matrix

class FullmodelE2E(nn.Module):
    def __init__(self, dim=256, n_classes=6, hidden_dim=128, tcn_layers=2, dropout=0.45):
        super().__init__()
        
        # 1. Encoders (Partono dai loro pesi pre-addestrati: ImageNet/MS-Celeb per DAN, base per WavLM)
        self.audio_encoder = WavLMAudioEncoder(out_dim=dim, n_tcn_layers=tcn_layers, dropout=dropout)
        self.video_encoder = DANVideoEncoder2(out_dim=dim, tcn_layers=tcn_layers, dropout=dropout)
        
        # 2. Token per la robustezza (Missing Modality)
        # Se in ROS non trovi la faccia, passerai video_mask=False e si attiverà questo token
        self.missing_audio_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.missing_video_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        # 3. Fusione Cross-Attention
        self.fusion = RobustCrossAttentionFusion(dim=dim, dropout=dropout)
        
        # 4. Classificatore Finale (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Aggiunto BatchNorm per stabilizzare l'E2E
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Parametri per lo smoothing in inferenza (ROS)
        self.ema_alpha = 0.3
        self.register_buffer('ema_logits', torch.zeros(1, n_classes))
        self.is_first_frame = True
        self.enable_ema = False

    def reset_ema(self):
        self.is_first_frame = True

    def forward(self, batch):
        audio = batch['audio']
        video = batch['video']
        
        # Estrazione Feature (B, Dim)
        za = self.audio_encoder(audio)
        zv = self.video_encoder(video)
        
        # --- LOGICA TOKEN (Applicata sia in Train che in Test/ROS) ---
        audio_mask = batch.get('audio_mask', None).to(za.device)
        video_mask = batch.get('video_mask', None).to(zv.device)
        
        if audio_mask is not None:
            # Sostituisce i sample mascherati con il missing_audio_token addestrabile
            za = torch.where(audio_mask.unsqueeze(1), za, self.missing_audio_token.expand_as(za))
            
        if video_mask is not None:
            # Sostituisce i video senza faccia (o mascherati) con il missing_video_token
            zv = torch.where(video_mask.unsqueeze(1), zv, self.missing_video_token.expand_as(zv))
        # -------------------------------------------------------------

        # Fusione e Classificazione
        z = self.fusion(za, zv)
        logits = self.classifier(z)
        
        # Smoothing per l'inferenza in ROS
        if not self.training and self.enable_ema:
            if self.is_first_frame:
                self.ema_logits = logits.detach()
                self.is_first_frame = False
            else:
                self.ema_logits = self.ema_alpha * logits.detach() + (1 - self.ema_alpha) * self.ema_logits
            return self.ema_logits
            
        return logits
    

def train_epoch_e2e(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch['label'])
        
        loss.backward()
        # Gradient Clipping: vitale nell'E2E per evitare che gradienti esplosivi dell'MLP distruggano DAN/WavLM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting E2E Training on {device}")
    
    # 1. Dataset (Usa l'Augmentor "sicuro" senza RandomCrop per i video)
    print("Loading Dataset...")
    aug = DataAugmentor()
    ds = CremaDSmartLoader(augmentor=aug, duration=3.0, modality_dropout_prob=0.15)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds, batch_size=32, from_idx=False)
    #train_dl = DataLoader(ds,32)
    
    # 2. Inizializzazione Modello
    model = FullmodelE2E(dim=256, n_classes=6, tcn_layers=2, dropout=0.45).to(device)
    
    # 3. Smistamento Chirugico dei Learning Rates (Il cuore dell'E2E)
    dan_backbone_params = []
    wavlm_lora_params = []
    scratch_params = [] # TCN, Attenzione, Pooling, MLP Classifier
    
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        if "video_encoder.backbone" in name:
            dan_backbone_params.append(param)
        elif "audio_encoder.model" in name:
            wavlm_lora_params.append(param)
        else:
            scratch_params.append(param)
            
    # L'MLP e le TCN imparano velocemente, DAN fa solo micro-aggiustamenti
    optimizer = optim.AdamW([
        {'params': dan_backbone_params, 'lr': 1e-5}, 
        {'params': wavlm_lora_params, 'lr': 1e-4},   
        {'params': scratch_params, 'lr': 1e-3}       
    ], weight_decay=1e-3)
    
    # 4. Loss & Scheduler
    # FocalLoss con pesi sbilanciati per distruggere il bias "Sad/Neutral"
    class_weights = torch.tensor([1.0, 1.0, 1.2, 1.0, 0.4, 2.5]).to(device)
    #loss_fn = FocalLoss(alpha=None, gamma=2.5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Scheduler: CosineAnnealing fa scendere morbidamente il LR in 60 epoche
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_f1 = 0.0
    save_path = "fullmodel_e2e_best.pt"
    
    # 5. Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch_e2e(model, train_dl, optimizer, device, loss_fn)
        val_loss, val_f1 = test_fullmodel(model, val_dl, device, loss_fn, dropout_prob=0.0)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  --> 🔥 Nuovo SOTA salvato! (F1: {val_f1:.4f})")
            
    # 6. Test Finale e Confusion Matrix
    print("\nTraining completato. Esecuzione Test finale...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_f1 = test_fullmodel(model, test_dl, device, loss_fn, dropout_prob=0.15)
    print(f"Test Finale (con 15% Modality Dropout) -> Loss: {test_loss:.4f} | F1: {test_f1:.4f}")
    
    # Salva la matrice
    save_confusion_matrix(model, test_dl, device, "cm_e2e_ce.png")
    print("Matrice di confusione salvata in cm_e2e.png")

if __name__ == '__main__':
    main()
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_2 import CremaDSmartLoader, DataAugmentor
from utils import make_weighted_loaders
from model4_2 import OptunaMultimodalEncoder, Fullmodel42
from mercl import MERCL_Loss, FocalLoss
from train_2 import pretrain_encoder4_acc, train_fullmodel4
from test import test_fullmodel, save_confusion_matrix, test_pretrained_encoder4
'''
def split_dataset(dataset, val_ratio=0.15, test_ratio=0.15):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split1 = int(n * (1 - val_ratio - test_ratio))
    split2 = int(n * (1 - test_ratio))
    train_idx = idx[:split1]
    val_idx = idx[split1:split2]
    test_idx = idx[split2:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
'''
def run_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load CREMA-D with new augmentation
    print("Loading Dataset & Augmentations...")
    augmentor = DataAugmentor()
    
    # Check if dataset exists
    if os.path.exists("/home/roano/standalone/crema-d"):
        has_dataset = True
        ds = CremaDSmartLoader(augmentor=augmentor, modality_dropout_prob=0.15)
        # Using make_weighted_loaders as in original train scripts for better class balance
        train_loader, val_loader, test_loader = make_weighted_loaders(ds, batch_size=32, from_idx=True)
        #train_loader = DataLoader(ds,32)
    else:
        print("Warning: CREMA-D dataset not found at /home/roano/standalone/crema-d.")
        print("Running in dry-run/mock mode to verify pipeline execution...")
        has_dataset = False
        
        class MockLoader:
            def __init__(self, n_batches=2):
                self.n_batches = n_batches
                self.dataset = [0] * n_batches  # mock dataset length
            def __iter__(self):
                for _ in range(self.n_batches):
                    yield {
                        'audio': torch.randn(2, 16000),
                        'video': torch.randn(2, 16, 3, 224, 224),
                        'label': torch.randint(0, 6, (2,)),
                        'audio_mask': torch.ones(2, dtype=torch.bool),
                        'video_mask': torch.ones(2, dtype=torch.bool)
                    }
            def __len__(self):
                return self.n_batches
                
        train_loader = MockLoader(2)
        val_loader = MockLoader(1)
        test_loader = MockLoader(1)
        
    backbones = ['mobilenet', 'dan']
    pool_type = 'att'
    
    for backbone in backbones:
        print(f"\n{'='*50}")
        print(f"PIPELINE FOR BACKBONE: {backbone.upper()}")
        print(f"{'='*50}")
        
        # ----------------------------------------------------------------------
        # Phase 1: Pretrain the OptunaMultimodalEncoder with MERCL_Loss
        # ----------------------------------------------------------------------
        print("\nPhase 1: Contrastive Pretraining (MERCL_Loss)")
        encoder = OptunaMultimodalEncoder(dim=256, tcn_layers=2, video_backbone=backbone, pool_type=pool_type, dropout=0.45).to(device)
        mercl_loss_fn = MERCL_Loss(temp=0.1, lambda_amcl=1.0, lambda_emcl=1.0, lambda_smcl=0.1).to(device)
        
        # Optimizer with differential learning rates for pretraining
        dan_params = []
        lora_params = []
        base_params = []
        for name,param in encoder.named_parameters():
            if not param.requires_grad: continue
            if "video_encoder.backbone" in name: dan_params.append(param)
            elif "audio_encoder.model" in name: lora_params.append(param)
            else: base_params.append(param)
            
        optimizer_pretrain = optim.AdamW([
            {'params': dan_params, 'lr': 1e-5}, 
            {'params': lora_params, 'lr': 1e-4}, 
            {'params': base_params, 'lr': 1e-3}  
        ], weight_decay=1e-4)
        
        #scheduler_pretrain = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretrain, mode='min', factor=0.5, patience=4)
        pretrain_epochs = 100 if has_dataset else 1
        scheduler_pretrain = optim.lr_scheduler.CosineAnnealingLR(optimizer_pretrain,pretrain_epochs,1e-6)
        
        
        patience_pretrain = 12
        wait = 0
        best_loss = float('inf')
        encoder_weights_path = f"pretrained_encoder_{backbone}_jules.pt"
        
        for epoch in range(pretrain_epochs):
            train_loss = pretrain_encoder4_acc(encoder, train_loader, optimizer_pretrain, device, mercl_loss_fn)
            val_loss = test_pretrained_encoder4(encoder, val_loader, device, mercl_loss_fn) if has_dataset else train_loss
            
            scheduler_pretrain.step()
            print(f"  Pretrain Epoch {epoch+1}/{pretrain_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                torch.save(encoder.state_dict(), encoder_weights_path)
            else:
                wait += 1
                if wait >= patience_pretrain and has_dataset:
                    print("  🛑 Early stopping triggered for pretraining")
                    break

        # ----------------------------------------------------------------------
        # Phase 2: Fine-Tuning Fusion and Classifier (FocalLoss)
        # ----------------------------------------------------------------------
        print("\nPhase 2: Fine-Tuning Fusion and Classifier (FocalLoss)")
        full_model = Fullmodel42(
            encoder_weights=encoder_weights_path if os.path.exists(encoder_weights_path) else None, 
            dim=256, 
            n_classes=6, 
            tcn_layers=2, 
            video_backbone=backbone, 
            pool_type=pool_type, 
            dropout=0.45
        ).to(device)
            
        focal_loss_fn = FocalLoss(gamma=2.0).to(device)
        '''
        # 2A. WARMUP (10 epochs, fusion and classifier only, unfreeze encoders later)
        warmup_epochs = 10 if has_dataset else 1
        optimizer_warmup = optim.AdamW(full_model.parameters(), lr=1e-3, weight_decay=1e-3)
        
        for p in full_model.audio_encoder.parameters(): p.requires_grad = False
        for p in full_model.video_encoder.parameters(): p.requires_grad = False
            
        for epoch in range(warmup_epochs):
            train_loss = train_fullmodel4(full_model, train_loader, optimizer_warmup, device, focal_loss_fn)
            _, val_f1 = test_fullmodel(full_model, val_loader, device, focal_loss_fn)
            print(f"  Warmup Epoch {epoch+1}/{warmup_epochs} - Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        # 2B. FULL FINETUNING (Differential LRs, Unfreeze Encoders)
        for p in full_model.audio_encoder.parameters(): p.requires_grad = True
        for p in full_model.video_encoder.parameters(): p.requires_grad = True
        '''
        optimizer_finetune = optim.AdamW([
            {'params': full_model.audio_encoder.parameters(), 'lr': 1e-5},
            {'params': full_model.video_encoder.parameters(), 'lr': 1e-5},
            {'params': full_model.fusion.parameters(), 'lr': 1e-3},
            {'params': full_model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-3)
        
        scheduler_finetune = optim.lr_scheduler.ReduceLROnPlateau(optimizer_finetune, mode='max', factor=0.2, patience=4, min_lr=1e-7)
        
        finetune_epochs = 60 if has_dataset else 1
        patience_finetune = 12
        wait = 0
        best_f1 = 0.0
        finetuned_weights_path = f"finetuned_fullmodel_{backbone}_jules.pt"
        
        for epoch in range(finetune_epochs):
            train_loss = train_fullmodel4(full_model, train_loader, optimizer_finetune, device, focal_loss_fn)
            _, val_f1 = test_fullmodel(full_model, val_loader, device, focal_loss_fn)
            
            scheduler_finetune.step(val_f1)
            print(f"  Finetune Epoch {epoch+1}/{finetune_epochs} - Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                torch.save(full_model.state_dict(), finetuned_weights_path)
            else:
                wait += 1
                if wait >= patience_finetune and has_dataset:
                    print("  🛑 Early stopping triggered for finetuning")
                    break
            
        # ----------------------------------------------------------------------
        # Phase 3: Metrics Evaluation
        # ----------------------------------------------------------------------
        print("\nPhase 3: Final Evaluation")
        if os.path.exists(finetuned_weights_path):
            full_model.load_state_dict(torch.load(finetuned_weights_path))
            
        test_loss, test_f1 = test_fullmodel(full_model, test_loader, device, focal_loss_fn, dropout_prob=0.15)
        print(f"  Final Test Loss (15% Modality Dropout): {test_loss:.4f} | Test F1 (macro): {test_f1:.4f}")
        
        # Save confusion matrix
        cm_path = f"confusion_matrix_{backbone}_jules.png"
        try:
            save_confusion_matrix(full_model, test_loader, device, cm_path)
            print(f"  Saved confusion matrix to {cm_path}")
        except Exception as e:
            print(f"  Could not save confusion matrix (expected in dry-run or missing GUI env): {e}")

if __name__ == "__main__":
    run_pipeline()
import torch
import torch.nn as nn
from dataset_2 import DataAugmentor, CremaDSmartLoader, ProcessedCremaDLoader, MultimodalAugmentor
from utils import make_weighted_loaders
from model4 import FullModel4, Fullmodel42
from train import train_fullmodel4, train_fullmodel_aug
from test import test_mlp_encoder
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mercl import FocalLoss
from sklearn.metrics import confusion_matrix, f1_score
import os

def main():
    warmup = True
    # hyperparameters
    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
    batch_size = 32
    latent_dim = 256
    epochs = 100
    head_warmup_epochs = 6  # train fusion+classifier only
    patience = 12
    best_f1 = 0.0
    wait = 0
    save_path = "/home/roano/standalone/models/final_model_3s.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloaders
    aug = MultimodalAugmentor()
    #ds = CremaDSmartLoader(duration=3.0,augmentor=None)
    ds = ProcessedCremaDLoader(mode='train',augmentor=aug,modality_dropout_prob=0.2,normalize=True)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,batch_size,from_idx=True)
    #train_dl = DataLoader(ds,batch_size)
    print('DONE MAKING LOADERS')

    # setup model
    model = Fullmodel42("/home/roano/standalone/models/final_encoder_3s.pt",latent_dim).to(device)
    model.load_state_dict(torch.load(save_path))

    # optimizer and loss
    #loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1,weight=torch.load('/home/roano/standalone/class_weights.pt'))
    loss_fn = FocalLoss(gamma=2.0)

    # training loop 
    train_loss_total = []
    val_loss_total = []
    
    if warmup:
        print("\n=== PHASE 1: HEAD-ONLY WARMUP (freeze encoders) ===\n")
        # ---- Head-only warmup: freeze encoders, train fusion+classifier ----
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        for p in model.video_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW([
            {'params': model.fusion.parameters(), 'lr': 1e-3},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-2)

        for e in range(head_warmup_epochs):
            train_loss = train_fullmodel_aug(model, train_dl, optimizer, device, loss_fn)
            train_loss_total.append(train_loss)

            val_f1, _, val_loss = test_mlp_encoder(model, val_dl, device, loss_fn)
            val_loss_total.append(val_loss)

            print(f"Head Warmup {e+1}/{head_warmup_epochs}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val F1 = {val_f1:.6f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✔ New best model saved ({val_f1:.6f})")

        print("\n=== PHASE 2: FINE-TUNE VIDEO HEAD + FUSION/CLASSIFIER (audio stays frozen) ===\n")
        # ---- Fine-tune fusion + classifier + video head; keep audio frozen ----
        for p in model.video_encoder.parameters():
            p.requires_grad = True
        for p in model.audio_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW([
            {'params': model.audio_encoder.parameters(), 'lr': 5e-5},
            {'params': model.video_encoder.parameters(), 'lr': 1e-4},
            {'params': model.fusion.parameters(), 'lr': 1e-3},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=6,
            min_lr=1e-7
        )

        for e in range(epochs):
            train_loss = train_fullmodel_aug(model, train_dl, optimizer, device, loss_fn)
            train_loss_total.append(train_loss)

            val_f1, _, val_loss = test_mlp_encoder(model, val_dl, device, loss_fn)
            val_loss_total.append(val_loss)

            scheduler.step(val_f1)

            print(f"Epoch {e+head_warmup_epochs+1}/{epochs+head_warmup_epochs}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val F1 = {val_f1:.6f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✔ New best model saved ({val_f1:.6f})")
            else:
                wait += 1
                print(f"  ⏳ No improvement ({wait}/{patience})")
                if wait >= patience:
                    print("🛑 Early stopping triggered")
                    break

    else:
        print("\n=== NO WARMUP: DIRECT TWO-STAGE TRAINING ===\n")
        # ---- Head-only warmup: freeze encoders, train fusion+classifier ----
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        for p in model.video_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW([
            {'params': model.fusion.parameters(), 'lr': 1e-3},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-2)

        for e in range(head_warmup_epochs):
            train_loss = train_fullmodel_aug(model, train_dl, optimizer, device, loss_fn)
            train_loss_total.append(train_loss)

            val_f1, _, val_loss = test_mlp_encoder(model, val_dl, device, loss_fn)
            val_loss_total.append(val_loss)

            print(f"Head Warmup {e+1}/{head_warmup_epochs}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val F1 = {val_f1:.6f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✔ New best model saved ({val_f1:.6f})")

        # ---- Fine-tune fusion + classifier + video head; keep audio frozen ----
        for p in model.video_encoder.parameters():
            p.requires_grad = True
        for p in model.audio_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW([
            {'params': model.audio_encoder.parameters(), 'lr': 5e-5},
            {'params': model.video_encoder.parameters(), 'lr': 1e-4},
            {'params': model.fusion.parameters(), 'lr': 1e-3},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=6,
            min_lr=1e-7
        )

        for e in range(epochs):
            train_loss = train_fullmodel_aug(model, train_dl, optimizer, device, loss_fn)
            train_loss_total.append(train_loss)

            val_f1, _, val_loss = test_mlp_encoder(model, val_dl, device, loss_fn)
            val_loss_total.append(val_loss)

            scheduler.step(val_f1)

            print(f"Epoch {e+head_warmup_epochs+1}/{epochs+head_warmup_epochs}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val F1 = {val_f1:.6f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                wait = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✔ New best model saved ({val_f1:.6f})")
            else:
                wait += 1
                print(f"  ⏳ No improvement ({wait}/{patience})")
                if wait >= patience:
                    print("🛑 Early stopping triggered")
                    break

    # ===== FINAL EVALUATION with triple modes =====
    print("\n=== FINAL EVALUATION: BOTH / AUDIO-ONLY / VIDEO-ONLY ===\n")
    
    def evaluate_with_mode(model, loader, device, loss_fn, mode, tag):
        """Evaluate model with specific modality configuration."""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Set masks and zero out modalities as needed
                if mode == 'audio_only':
                    batch['video'] = torch.zeros_like(batch['video'])
                    batch['video_mask'] = torch.zeros(batch['audio'].size(0), dtype=torch.bool, device=device)
                elif mode == 'video_only':
                    batch['audio'] = torch.zeros_like(batch['audio'])
                    batch['audio_mask'] = torch.zeros(batch['audio'].size(0), dtype=torch.bool, device=device)
                
                logits = model(batch)
                loss = loss_fn(logits, batch['label'])
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Compute metrics
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        loss_avg = total_loss / len(loader)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({tag}) – F1-macro={f1:.3f}")
        plt.tight_layout()
        plt.savefig(f"/home/roano/standalone/figures/confusion_matrix_final_model_3s_{tag}.png")
        plt.close()
        
        return f1, loss_avg

    # Load best model and evaluate on all three modes
    model.load_state_dict(torch.load(save_path))
    
    f1_both, loss_both = evaluate_with_mode(model, test_dl, device, loss_fn, mode='both', tag='both')
    f1_audio, loss_audio = evaluate_with_mode(model, test_dl, device, loss_fn, mode='audio_only', tag='audio')
    f1_video, loss_video = evaluate_with_mode(model, test_dl, device, loss_fn, mode='video_only', tag='video')
    
    print(f"\nFinal F1 Results:")
    print(f"  Both modalities: {f1_both:.3f} (loss: {loss_both:.6f})")
    print(f"  Audio-only:      {f1_audio:.3f} (loss: {loss_audio:.6f})")
    print(f"  Video-only:      {f1_video:.3f} (loss: {loss_video:.6f})")

    # saving train/test graphs
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(val_loss_total, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss - Full Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/roano/standalone/figures', "loss_curve_final_model_3s.png"))
    plt.close()
    
    print('\n✅ DONE!')




if __name__ == '__main__':
    main()
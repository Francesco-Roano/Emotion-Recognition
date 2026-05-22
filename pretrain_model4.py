import torch
from dataset_2 import CremaDSmartLoader, DataAugmentor, ProcessedCremaDLoader, MultimodalAugmentor
from utils import make_weighted_loaders, speaker_disjoint_split
from model4 import MultimodalEncoder2
from mercl import MERCL_Loss
from train import pretrain_encoder4,pretrain_encoder4_acc
from test import test_pretrained_encoder4
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def main():
    
    # hyperparameters
    batch_size = 32
    latent_dim = 256
    epochs = 100
    patience = 12
    best_loss = float('inf')
    wait = 0
    save_path = "/home/roano/standalone/models/pretrained_encoder4_landmark_aug_norm_3s.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloaders
    aug = MultimodalAugmentor(p_audio=0.7, p_video=0.9, sample_rate=16000)
    #ds = CremaDSmartLoader(augmentor=None,duration=3.0)
    ds = ProcessedCremaDLoader(
        #data_dir='/home/roano/standalone/data/cremad_smart_processed_1s',
        augmentor=aug,
        normalize=True
    )
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,batch_size,from_idx=True)
    #train_dl = DataLoader(ds,batch_size)
    #train_dl,val_dl,test_dl = speaker_disjoint_split(ds,1,from_idx=True)
    print('DONE MAKING LOADERS')

    # setup model
    encoder = MultimodalEncoder2(audio_dropout=0.5).to(device)
    #encoder.load_state_dict(torch.load(save_path))
    
    # optimizer and loss
    #optimizer = torch.optim.AdamW(encoder.parameters(),lr=1e-4,weight_decay=1e-4)
    dan_params = []
    lora_params = []
    base_params = []
    proj_params = []
    for name,param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        if "video_encoder.backbone" in name:
            dan_params.append(param)
        elif "audio_encoder.model" in name:
            lora_params.append(param)
        elif "audio_proj" in name or "video_proj" in name:
            proj_params.append(param)
        else:
            base_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': dan_params, 'lr': 1e-5}, 
        {'params': lora_params, 'lr': 1e-4}, 
        {'params': base_params, 'lr': 1e-3}, 
        {'params': proj_params, 'lr': 5e-4}  
    ], weight_decay=1e-2)
    loss_fn = MERCL_Loss(
        temp=0.2,
        lambda_amcl=1.0,
        lambda_emcl=1.0,
        lambda_smcl=0.05,
        alpha=0.9,
        hnm=True,
        weighted=False
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

    # training loop
    train_loss_total = []
    val_loss_total = []
    for e in range(epochs):

        train_loss = pretrain_encoder4_acc(encoder,train_dl,optimizer,device,loss_fn)
        train_loss_total.append(train_loss)

        val_loss = test_pretrained_encoder4(encoder,val_dl,device,loss_fn)
        val_loss_total.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {e+1}/{epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f}")

        # early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(encoder.state_dict(),save_path)
            print(f"  ✔ New best model saved ({val_loss:.6f})")
        else:
            wait += 1
            print(f"  ⏳ No improvement ({wait}/{patience})")
            if wait >= patience:
                print("🛑 Early stopping triggered")
                break

    # final test
    test_loss = test_pretrained_encoder4(encoder,test_dl,device,loss_fn)

    # saving train/test graphs
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(val_loss_total, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss - Encoder4 Pretrain - Test loss: "+str(test_loss))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/roano/standalone/figures', "loss_curve_encoder4_pretrain_3s_landmark_aug_norm_2.png"))
    plt.close()


if __name__ == '__main__':
    main()
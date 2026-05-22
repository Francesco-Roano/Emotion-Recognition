from dataset import CremaEncodedDataset, CremaDLoader
from model2 import EncodingBlock, CrossAttention, Fusion, MlpClassifier, FullModel, EncodingBlockNoCompression, StableCrossAttention
import torch
import torch.nn as nn
from train import train_fullmodel,overfitting_training
from test import test_fullmodel
import os
from utils import split_loaders, compute_class_weights
import matplotlib.pyplot as plt

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # hyperparameters
    latent_dim = 512
    post_att_dim = 126
    attention_heads = 2
    batch_size = 8
    lr = 1e-4
    epochs = 10
    patience=8
    min_delta=1e-4
    best_f1 = float("inf")
    wait = 0

    # load crema-d dataset
    ds = CremaEncodedDataset()
    train_loader, test_loader = split_loaders(ds,batch_size)  
    
    # construct model
    encoder = EncodingBlock(latent_dim)
    gating = StableCrossAttention(latent_dim=512,heads=attention_heads)
    fusion = Fusion(latent_dim=latent_dim,out_dim=post_att_dim,gating=gating)
    classifier = MlpClassifier(in_dim=post_att_dim,n_classes=6)
    model = FullModel(encoder,fusion,classifier).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.load("/home/roano/standalone/class_weights.pt").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # training
    train_loss_total = []
    test_loss_total = []
    f1_total = []
    for e in range(epochs):
        train_loss,_ = train_fullmodel(model,train_loader,optimizer,device,criterion)
        train_loss_total.append(train_loss)
        
        test_loss,f1 = test_fullmodel(model,test_loader,device,criterion)
        test_loss_total.append(test_loss)
        f1_total.append(f1)
        print(f"Epoch {e+1}/{epochs} - Train loss: {train_loss:.6f} - Test loss: {test_loss:.6f} - F1-macro: {f1:.2f}")

        #early stopping check
        if f1 < best_f1 - min_delta:
            best_f1 = f1
            wait = 0

            torch.save(model.state_dict(), "/home/roano/standalone/models/fullmodel2.pt")

            print(f"  ✔ New best model saved ({f1:.6f})")

        else:
            wait += 1
            print(f"  ⏳ No improvement ({wait}/{patience})")

            if wait >= patience:
                print("🛑 Early stopping triggered")
                break

    # saving results
    """
    save_path = "/home/roano/standalone/models/fullmodel.pt"
    print("[DEBUG] Trying to save here:", save_path)
    try:
        torch.save(model.state_dict(), save_path)
        print("[DEBUG] Saved!")
    except Exception as e:
        print("[DEBUG] ERROR during saving:", e)
    """

    # saving train/test graphs
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(test_loss_total, label="Validation")
    plt.plot(f1_total, label="F1-macro")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/roano/standalone/figures', "loss_curve_full.png"))
    plt.close()
    
if __name__ == "__main__":
    main()


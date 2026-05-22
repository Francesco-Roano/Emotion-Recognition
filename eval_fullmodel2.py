from dataset import CremaEncodedDataset
from model2 import EncodingBlock, Fusion, FullModel, SmallClassifier, EncodingBlockNoCompression, StableCrossAttention, MlpClassifier
import torch
import torch.nn as nn
from train import train_fullmodel
from test import test_fullmodel, save_confusion_matrix
import os
from utils import train_val_test, make_weighted_loaders
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
    epochs = 30
    patience = 5
    min_delta=1e-4
    best_f1 = 0.0
    wait = 0
    save_path = "/home/roano/standalone/models/fullmodel_newloss.pt"

    # load crema-d dataset
    ds = CremaEncodedDataset()
    train_loader, val_loader, test_loader = train_val_test(ds,batch_size,from_idx=True)
    
    # construct model
    encoder = EncodingBlock(latent_dim=latent_dim)
    gating = StableCrossAttention(latent_dim=512,heads=attention_heads)
    fusion = Fusion(latent_dim=latent_dim,out_dim=1024,gating=None)
    #classifier = MlpClassifier(in_dim=512,n_classes=6)
    classifier = SmallClassifier(6,1024)
    model = FullModel(encoder,fusion,classifier).to(device)
    #model.load_state_dict(torch.load(save_path))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.load("/home/roano/standalone/class_weights.pt").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # training
    train_loss_total = []
    val_loss_total = []
    f1_total = []
    for e in range(epochs):
        train_loss,_ = train_fullmodel(model,train_loader,optimizer,device,criterion)
        train_loss_total.append(train_loss)
        
        val_loss,f1 = test_fullmodel(model,val_loader,device,criterion)
        val_loss_total.append(val_loss)
        f1_total.append(f1)
        print(f"Epoch {e+1}/{epochs} - Train loss: {train_loss:.6f} - Validation loss: {val_loss:.6f} - F1-macro: {f1:.2f}")

        #early stopping check
        if f1 > best_f1 - min_delta:
            best_f1 = f1
            wait = 0

            torch.save(model.state_dict(), save_path)

            print(f"  ✔ New best model saved ({f1:.6f})")

        else:
            wait += 1
            print(f"  ⏳ No improvement ({wait}/{patience})")

            if wait >= patience:
                print("🛑 Early stopping triggered")
                break

    # saving train/test graphs
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(val_loss_total, label="Validation")
    plt.plot(f1_total, label="F1-macro")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/roano/standalone/figures', "loss_curve_full_newloss.png"))
    plt.close()

    # final evaluation
    save_confusion_matrix(model,test_loader,device,"/home/roano/standalone/figures/confusion_matrix_newloss.png")

    print("DONE")
    
if __name__ == "__main__":
    main()


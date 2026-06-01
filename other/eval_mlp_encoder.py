import torch
from torch.utils.data import DataLoader
from dataset import CremaDSmartLoader
from utils import make_weighted_loaders, extract_embeddings_attention
from model3 import AttentionEncoder, MlpFusion
from train import train_cross_attention, train_mlp_encoder
from test import test_cross_attention, test_mlp_encoder
from utils import Loss_3
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import seaborn as sns

def main():

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
    save_path = '/home/roano/standalone/models/model3_mlp_enc_05_1.pt'
    epochs = 20
    best_f1 = 0.0
    wait = 0
    patience = 5
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CremaDSmartLoader(n_frames=8)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,32,from_idx=True)
    #train_dl = DataLoader(ds,batch_size=32)
    print('DONE MAKING LOADERS!')

    model = MlpFusion().to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_total = []
    val_f1_total = []

    for e in range(epochs):
        train_loss = train_mlp_encoder(model,train_dl,optimizer,device,loss_fn)
        train_loss_total.append(train_loss)

        val_f1,_ = test_mlp_encoder(model,val_dl,device)
        val_f1_total.append(val_f1)

        print(f"Epoch {e+1}/{epochs}: Train Loss = {train_loss:.6f} | Val F1 = {val_f1:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(),save_path)
            print(f"  ✔ New best model saved ({val_f1:.6f})")
        else:
            wait += 1
            print(f"  ⏳ No improvement ({wait}/{patience})")
            if wait >= patience:
                print("🛑 Early stopping triggered")
                break
    
    # final test
    f1,cm = test_mlp_encoder(model,test_dl,device)
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
    plt.title(f"MLP classifier Confusion Matrix – F1-macro={f1:.3f}")
    plt.tight_layout()
    plt.savefig("/home/roano/standalone/figures/confusion_matrix_mlp_model3_05_1.png")
    plt.close()

    print('DONE!')

if __name__ == '__main__':
    main()
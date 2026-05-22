import torch
from torch.utils.data import DataLoader
from dataset import CremaDSmartLoader
from utils import make_weighted_loaders, extract_embeddings_attention
from model3 import AttentionEncoder
from train import train_cross_attention
from test import test_cross_attention
from utils import Loss_3
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def main():

    save_path = '/home/roano/standalone/models/model3_attention_enc_05_1_margin.pt'
    epochs = 20
    best_loss = float('inf')
    wait = 0
    patience = 5
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CremaDSmartLoader(n_frames=8)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,32,from_idx=True)
    #train_dl = DataLoader(ds,batch_size=32)
    print('DONE MAKING LOADERS!')

    model = AttentionEncoder(512,6).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr)
    loss_fn = Loss_3(device,sup_weight=0.5,ce_weight=1.0)

    train_loss_total = []
    val_loss_total = []

    for e in range(epochs):
        train_loss = train_cross_attention(model,train_dl,optimizer,device,loss_fn)
        train_loss_total.append(train_loss)

        val_loss = test_cross_attention(model,val_dl,device,loss_fn)
        val_loss_total.append(val_loss)

        print(f"Epoch {e+1}/{epochs}: Train Loss = {train_loss:.6f} | Val F1 = {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(),save_path)
            print(f"  ✔ New best model saved ({val_loss:.6f})")
        else:
            wait += 1
            print(f"  ⏳ No improvement ({wait}/{patience})")
            if wait >= patience:
                print("🛑 Early stopping triggered")
                break
    
    # final test
    train_feats, train_labels = extract_embeddings_attention(model,train_dl,device)
    test_feats, test_labels = extract_embeddings_attention(model,test_dl,device)
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats = scaler.fit_transform(test_feats)
    knn = KNeighborsClassifier(
            n_neighbors=21,
            metric='cosine',
            weights='distance'
        )
    knn.fit(train_feats,train_labels)
    preds = knn.predict(test_feats)

    f1 = f1_score(test_labels,preds, average='macro')
    print('Final F1 score: ',f1)

    print('DONE!')

if __name__ == '__main__':
    main()
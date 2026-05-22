import torch
from dataset import CremaDSmartLoader
from utils import make_weighted_loaders, Loss_3, extract_embeddings_onlyvideo
from model3 import VideoEncoder
from train import train_video_autoencoder_3
from test import test_video_autoencoder_3
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def main():
    
    # hyperparameters
    batch_size = 32
    latent_dim = 512
    hidden_dim = 512
    lr = 4e-4
    epochs = 20
    patience = 5
    save_path = "/home/roano/standalone/models/model3_video_1_1.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = float('inf')
    wait = 0

    # setup dataset
    ds = CremaDSmartLoader(n_frames = 8)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,batch_size,from_idx=True)
    #train_dl = DataLoader(ds,batch_size)
    print('DONE MAKING LOADERS')

    # setup model
    encoder = VideoEncoder().to(device)
    #encoder.load_state_dict(torch.load(save_path))
    
    # optimizer and loss
    optimizer = torch.optim.Adam(encoder.parameters(),lr=lr)
    loss_fn = Loss_3(temperature=0.07,sup_weight=0.5,ce_weight=1.0)

    # training loop
    train_loss_total = []
    val_loss_total = []
    for e in range(epochs):

        train_loss = train_video_autoencoder_3(encoder,train_dl,optimizer,loss_fn,device)
        train_loss_total.append(train_loss)

        val_loss = test_video_autoencoder_3(encoder,val_dl,loss_fn,device)
        val_loss_total.append(val_loss)

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
    train_feats, train_labels,_ = extract_embeddings_onlyvideo(encoder,train_dl,device)
    test_feats, test_labels,_ = extract_embeddings_onlyvideo(encoder,test_dl,device)
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


if __name__=='__main__':
    main()
import torch
from utils import split_loaders
from dataset import CremaDLoader, CremaEncodedDataset
from model2 import TimeSformerCNNDecoder, TimeSformerCNNEncoder, TimeSformerEmbedding, VideoAutoencoder
from train import train_video_autoencoder
from test import test_video_autoencoder
import os
import matplotlib.pyplot as plt


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    batch_size = 1
    latent_dim = 512
    lr = 4.6e-4
    epochs = 60 #41

    # load crema-d dataset
    ds = CremaEncodedDataset()
    train_loader, test_loader = split_loaders(ds,batch_size)


    # construct model
    encoder = TimeSformerCNNEncoder(latent_dim=latent_dim)
    decoder = TimeSformerCNNDecoder(latent_dim=latent_dim)
    model = VideoAutoencoder(encoder,decoder).to(device)
    model.load_state_dict(torch.load("/home/roano/standalone/models/video_autoencoder.pt"))

    # choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    train_loss_total = []
    test_loss_total = []
    for e in range(epochs):

        train_loss,_,_ = train_video_autoencoder(embed=None,model=model,loader=train_loader,optimizer=optimizer,device=device,loss_mode='mse+cos')
        train_loss_total.append(train_loss)

        test_loss = test_video_autoencoder(model,test_loader,device)
        test_loss_total.append(test_loss)

        print(f"Epoch {e+1}/{epochs} - Train loss: {train_loss:.6f} - Test loss: {test_loss:.6f}")

    # saving results
    save_path = "/home/roano/standalone/models/video_autoencoder.pt"
    print("[DEBUG] Trying to save here:", save_path)
    try:
        torch.save(model.state_dict(), save_path)
        print("[DEBUG] Saved!")
    except Exception as e:
        print("[DEBUG] ERROR during saving:", e)
    
    # saving train/test graphs
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(test_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/home/roano/standalone/figures', "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()

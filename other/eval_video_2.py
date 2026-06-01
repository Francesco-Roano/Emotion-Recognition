from dataset import CremaEncodedDataset
import torch
from utils import make_weighted_loaders, SupConLoss, NewLoss
from model2 import TimeSformerCNNEncoder, TimeSformerCNNDecoder, VideoAutoencoder, SupVideoAutoencoder
from model3 import ConvVideoEncoder2,ConvVideoDecoder2
from train import train_video_autoencoder_2
from test import test_video_autoencoder_2
import matplotlib.pyplot as plt

def main():

    # hyperparameters
    batch_size = 16
    latent_dim = 512
    lr = 4.6e-4
    epochs = 30
    patience = 8
    best_loss = float('inf')
    wait = 0
    save_path = "/home/roano/standalone/models/video_autoencoder_optuna_smart.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloaders
    ds = CremaEncodedDataset(smart=True)
    train_dl, val_dl, test_dl = make_weighted_loaders(ds,batch_size=batch_size)

    # construct model
    encoder = TimeSformerCNNEncoder(latent_dim)
    decoder = TimeSformerCNNDecoder(latent_dim)
    model = SupVideoAutoencoder(encoder,decoder,latent_dim,6).to(device)
    #model.load_state_dict(torch.load(save_path))

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = NewLoss(temperature=0.4,cos_weight=0.2,sup_weight=0.1,ce_weight=0.04)

    # training loop
    train_loss_total = []
    val_loss_total = []
    for e in range(epochs):

        train_loss = train_video_autoencoder_2(model,train_dl,optimizer,loss_fn,device)
        train_loss_total.append(train_loss)

        val_loss = test_video_autoencoder_2(model,val_dl,loss_fn,device)
        val_loss_total.append(val_loss)

        print(f"Epoch {e+1}/{epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f}")

        # early stopping check
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
    test_loss = test_video_autoencoder_2(model,test_dl,loss_fn,device)

    # save graph
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_total, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss. Final test loss = "+str(test_loss))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/roano/standalone/figures/loss_curve_video_v2.png')
    plt.close()

if __name__ == '__main__':
    main()
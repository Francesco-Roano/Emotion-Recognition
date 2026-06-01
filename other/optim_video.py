import optuna
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import CremaDLoader, CremaEncodedDataset
from model2 import TimeSformerEmbedding, TimeSformerCNNEncoder, TimeSformerCNNDecoder, VideoAutoencoder
from train import train_video_autoencoder
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Optuna Video Optimization")
    parser.add_argument("--data_path", type=str, default='/home/roano/standalone/crema-d-encoded', help="Path to Crema-D root")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--storage", type=str, default="video_optuna_results.csv")
    return parser.parse_args()

class VideoObjective:
    def __init__(self, args, device):
        self.args = args
        #self.embedding_model = embedding_model
        self.device = device
        self.best_loss = float('inf')

    def __call__(self, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4]) 
        latent_dim = trial.suggest_categorical("latent_dim", [256, 512, 1024])
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        
        # setup dataset
        ds = CremaEncodedDataset()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

        # model setup
        encoder = TimeSformerCNNEncoder(latent_dim=latent_dim)
        decoder = TimeSformerCNNDecoder(latent_dim=latent_dim)
        model = VideoAutoencoder(encoder, decoder).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # training loop
        final_loss = 0
        for epoch in tqdm(range(self.args.epochs)):
            epoch_loss, _ , _= train_video_autoencoder(
                None, model, loader, optimizer, 
                device=self.device, p=1.0, loss_mode='mse+cos'
            )
            
            #if epoch==0:
            #    print(f"Timesformer embedding took {t*1000:.2f} ms")

            # pruning
            trial.report(epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            final_loss = epoch_loss

        # save model if it's the best one
        if final_loss < self.best_loss:
            self.best_loss = final_loss
            print(f"--> New Best Video Model found! Loss: {self.best_loss:.6f}")
            torch.save(model.state_dict(), "best_video_autoencoder.pt")

        return final_loss

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load pretrained model
    #embedder = TimeSformerEmbedding().to(device) 

    objective = VideoObjective(args, device)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df = study.trials_dataframe()
    df.to_csv(args.storage, index=False)
    print(f"Results saved to {args.storage}")

if __name__ == "__main__":
    main()
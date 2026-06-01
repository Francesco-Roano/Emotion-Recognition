import optuna,torch
from model2 import ConvAudioEncoder,ConvAudioDecoder,SupAudioAutoencoder,TimeSformerCNNEncoder,TimeSformerCNNDecoder,SupVideoAutoencoder
from dataset import CremaEncodedDataset
from utils import make_weighted_loaders, NewLoss, extract_embeddings_onlyaudio,extract_embeddings_onlyvideo
from train import train_audio_autoencoder_2,train_video_autoencoder_2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

def save_best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        print(f"Saving best model from trial {trial.number}")
        torch.save(
            {
                "audio": trial.user_attrs["audio_state"],
                "video": trial.user_attrs["video_state"],
                "params": trial.params,
                "f1": trial.value
            },
            "/home/roano/standalone/models/best_model.pt"
        )

def objective(trial):

    # ---- hyperparameters ----
    latent_dim = 512
    batch_size = 32
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    cos_weight = trial.suggest_float("cos_weight", 0.0, 0.5)
    sup_weight = trial.suggest_float("sup_weight", 0.1, 1.0)
    ce_weight  = trial.suggest_float("ce_weight", 0.0, 0.5)
    temperature = trial.suggest_float("temperature", 0.05, 0.5)
    
    print(f"===========\nTrying:\n\tlatent_dim: {latent_dim}\n\tbatch_size: {batch_size}\n\tlr: {lr:.3f}\n\tcos_weight: {cos_weight:.3f}\n\tsup_weight: {sup_weight:.3f}\n\tce_weight: {ce_weight:.3f}\n\ttemperature: {temperature:.3f}\n")

    # ---- models ----
    ae = ConvAudioEncoder(latent_dim)
    ad = ConvAudioDecoder(latent_dim)
    audio_model = SupAudioAutoencoder(ae,ad,latent_dim,6).to(DEVICE)

    ve = TimeSformerCNNEncoder(latent_dim)
    vd = TimeSformerCNNDecoder(latent_dim)
    video_model = SupVideoAutoencoder(ve,vd,latent_dim,6).to(DEVICE)

    # ---- dataset ----
    dataset = CremaEncodedDataset()
    train_dl, val_dl, _ = make_weighted_loaders(dataset,batch_size,from_idx=True)

    # ---- loss & optim ----
    loss_fn = NewLoss(temperature,cos_weight,sup_weight,ce_weight)
    audio_opt = torch.optim.Adam(audio_model.parameters(), lr=lr)
    video_opt = torch.optim.Adam(video_model.parameters(), lr=lr)

    max_epochs = 15
    eval_epochs = [5, 10, 15]  # validation points

    best_f1 = 0.0

    for epoch in range(1, max_epochs + 1):

        train_audio_autoencoder_2(audio_model,train_dl,audio_opt,loss_fn,DEVICE)

        train_video_autoencoder_2(video_model,train_dl,video_opt,loss_fn,DEVICE)



        # ---- KNN eval + pruning ----
        if epoch in eval_epochs:

            train_audio_feats, train_labels = extract_embeddings_onlyaudio(audio_model,train_dl,DEVICE)
            train_video_feats, _ = extract_embeddings_onlyvideo(video_model,train_dl,DEVICE)
            val_audio_feats, val_labels = extract_embeddings_onlyaudio(audio_model,val_dl,DEVICE)
            val_video_feats, _ = extract_embeddings_onlyvideo(video_model,val_dl,DEVICE)

            train_feats = np.concatenate([train_audio_feats, train_video_feats], axis=1)
            val_feats = np.concatenate([val_audio_feats, val_video_feats], axis=1)

            scaler = StandardScaler()
            train_feats = scaler.fit_transform(train_feats)
            val_feats = scaler.fit_transform(val_feats)

            knn = KNeighborsClassifier(
                n_neighbors=21,
                metric='cosine',
                weights='distance'
            )

            knn.fit(train_feats,train_labels)
            preds = knn.predict(val_feats)
            f1 = f1_score(val_labels,preds,average='macro')
            
            print(f"VALIDATION STEP:\n\tf1-score: {f1:.3f}")

            best_f1 = max(best_f1, f1)

            # PRUNING HOOK
            trial.report(best_f1, step=epoch)

            if trial.should_prune():
                results.append({
                    "trial": trial.number,
                    "f1": f1,
                    "latent_dim": latent_dim,
                    "batch_size": batch_size,
                    "lr": lr,
                    "cos_weight": cos_weight,
                    "sup_weight": sup_weight,
                    "ce_weight": ce_weight,
                    "temperature": temperature
                })
                print("\nTRIAL PRUNED !!!\n")
                raise optuna.exceptions.TrialPruned()

    # ---- save best model ----
    results.append({
        "trial": trial.number,
        "f1": f1,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "lr": lr,
        "cos_weight": cos_weight,
        "sup_weight": sup_weight,
        "ce_weight": ce_weight,
        "temperature": temperature
    })

    return best_f1

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=30)

    df = pd.DataFrame(results)
    df.to_csv("/home/roano/encoder_optuna_results_2.csv", index=False)

    print("\n===========\nBest trial:")
    print(study.best_trial.params)
    print("Best F1:", study.best_value)
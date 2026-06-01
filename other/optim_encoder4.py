import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import os
from dataset import CremaDSmartLoader,DataAugmentor
from utils import make_weighted_loaders, knn_validation
from model4 import OptunaMultimodalEncoder
from mercl import MERCL_Loss
from train import pretrain_encoder4_acc
from test import test_pretrained_encoder4
from torch.utils.data import DataLoader

CSV_OUTPUT_PATH = "optuna_results_encoder4_2.csv"

def save_results_callback(study,frozen_trial):
    try:
        df = study.trials_dataframe()
        df.to_csv(CSV_OUTPUT_PATH,index=False)
        print('RESULTS SAVED!!!')
    except Exception as e:
        print(f'ERROR DURING RESULTS SAVING: {e}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

aug = DataAugmentor()
ds = CremaDSmartLoader(duration=3.0,augmentor=aug)
train_dl, val_dl, _ = make_weighted_loaders(ds,32,from_idx=True)
#train_dl = DataLoader(ds,32)
print('DONE MAKING LOADERS')

def objective(trial):
    video_backbone = trial.suggest_categorical('video_backbone',['dan','mobilenet'])
    pool_type = trial.suggest_categorical('pool_type',['mean','att','gru'])
    dim = trial.suggest_categorical('dim',[128,256,512])
    tcn_layers = trial.suggest_categorical('tcn_layers',[2,3,4])
    dropout = trial.suggest_float('dropout',0.3,0.6)
    temp = trial.suggest_float('temp',0.05,0.15)
    lambda_amcl = trial.suggest_float('lambda_amcl',0.5,1.5)
    lambda_emcl = trial.suggest_float('lambda_emcl',0.8,1.5)
    hnm = trial.suggest_categorical('hnm',['True','False'])
    weighted = trial.suggest_categorical('weighted',['True','False'])

    try:
        encoder = OptunaMultimodalEncoder(dim,tcn_layers,video_backbone,pool_type,dropout).to(device)
    except Exception as e:
        print(f'Failed configuration: {e}')
        raise optuna.exceptions.TrialPruned()
    
    finetune_params, lora_params, base_params = [], [], []
    for name, param in encoder.named_parameters():
        if not param.requires_grad: continue
        if "video_encoder.backbone" in name: finetune_params.append(param)
        elif "audio_encoder.model" in name: lora_params.append(param)
        else: base_params.append(param)   
    optimizer = AdamW([
        {'params': finetune_params, 'lr': 1e-5},
        {'params': lora_params, 'lr': 1e-4},
        {'params': base_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    loss_fn = MERCL_Loss(temp,lambda_amcl,lambda_emcl,hnm=hnm,weighted=weighted)

    for epoch in range(7):
        _ = pretrain_encoder4_acc(encoder,train_dl,optimizer,device,loss_fn)
        val_loss = test_pretrained_encoder4(encoder,val_dl,device,loss_fn)
        trial.report(val_loss,epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    f1 = knn_validation(encoder,train_dl,val_dl,device)
    return f1

if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=2)

    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name='encoder4_optim'
    )

    study.optimize(objective,n_trials=50,callbacks=[save_results_callback])

    print('FINISHED')
    print(f'Best Trial: ',study.best_trial.params)

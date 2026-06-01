from utils import embedding_similarity_loss, cross_entropy_loss, softF1Loss
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
import seaborn as sns
import os
import torch.nn.functional as F

def test_video_autoencoder(model,loader,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['video'].to(device)
        y,_ = model(x)
        loss = embedding_similarity_loss(x,y)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_video_autoencoder_2(model,loader,loss_fn,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['video'].to(device)
        labels = batch['label'].to(device)
        x_rec,z,logits = model(x)
        loss = loss_fn(x,x_rec,z,labels,logits)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_video_autoencoder_3(model,loader,loss_fn,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['video'].to(device)
        labels = batch['label'].to(device)
        z,logits = model(x)
        loss = loss_fn(z,logits,labels)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_audio_autoencoder(model,loader,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['audio'].to(device)
        y,_ = model(x)
        loss = embedding_similarity_loss(x,y)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_audio_autoencoder_2(model,loader,loss_fn,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['audio'].to(device)
        labels = batch['label'].to(device)
        x_rec,z,logits = model(x)
        loss = loss_fn(x,x_rec,z,labels,logits)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_audio_autoencoder_3(model,loader,loss_fn,device):
    model.eval()
    test_loss = 0.0
    for batch in loader:
        x = batch['audio'].to(device)
        labels = batch['label'].to(device)
        z,logits = model(x)
        #loss = loss_fn(z,logits,labels)
        loss = loss_fn(logits,labels)
        test_loss += loss.item()
    return test_loss/len(loader)

def test_fullmodel(model,loader,device,criterion,dropout_prob=0.0):
    model.eval()
    test_loss = 0.0
    f1_avg = 0.0
    
    # Store predictions for sklearn macro F1 to be more precise over full dataset
    all_preds = []
    all_labels = []
    
    if isinstance(model, torch.nn.Module) and hasattr(model, 'reset_ema'):
        model.reset_ema()
        
    for batch in loader:
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        batch["label"] = batch["label"].to(device)
        
        if "audio_mask" in batch:
            batch["audio_mask"] = batch["audio_mask"].to(device)
        if "video_mask" in batch:
            batch["video_mask"] = batch["video_mask"].to(device)
            
        # Apply dropout at test time for robustness evaluation
        if dropout_prob > 0.0:
            B = batch["audio"].shape[0]
            audio_mask = torch.ones(B, dtype=torch.bool, device=device)
            video_mask = torch.ones(B, dtype=torch.bool, device=device)
            
            for i in range(B):
                p = torch.rand(1).item()
                if p < dropout_prob:
                    batch["audio"][i] = 0.0
                    audio_mask[i] = False
                elif p < 2 * dropout_prob:
                    batch["video"][i] = 0.0
                    video_mask[i] = False
            batch["audio_mask"] = audio_mask
            batch["video_mask"] = video_mask
        
        label = batch['label']
        with torch.no_grad():
            # Handle models that might return only logits (like Fullmodel42)
            out = model(batch)
            if isinstance(out, tuple):
                probs, logits = out
            else:
                logits = out
                probs = F.softmax(logits, dim=-1)
                
        loss = criterion(logits,label)
        test_loss += loss.item()
        _,f1_macro = softF1Loss(probs,label)
        f1_avg += f1_macro.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
    f1_macro_exact = f1_score(all_labels, all_preds, average='macro')
    return test_loss/len(loader), f1_macro_exact

def save_confusion_matrix(model,test_loader,device,save_path):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch["audio"] = batch["audio"].to(device)
            batch["video"] = batch["video"].to(device)
            batch["label"] = batch["label"].to(device)
            labels = batch["label"]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"],
        yticklabels=["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return

def test_cross_attention(model,loader,device,loss_fn):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        labels = batch['label'].to(device)
        z,logits = model(batch)
        loss = loss_fn(z,logits,labels)
        total_loss += loss.item()
    return total_loss / len(loader)

def test_pretrained_encoder4(model,loader,device,loss_fn):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        labels = batch['label']
        with torch.no_grad():
            za,zv = model(batch)
        loss,_ = loss_fn(za,zv,labels)
        total_loss += loss.item()
    return total_loss/len(loader)

def test_mlp_encoder(model,loader,device,loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    for batch in loader:
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        labels = batch['label']
        with torch.no_grad():
            logits = model(batch)
        loss = loss_fn(logits,labels.to(device))
        total_loss += loss.item()
        preds = torch.argmax(logits,dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].numpy())
    f1 = f1_score(all_labels,all_preds,average='macro')
    cm = confusion_matrix(all_labels,all_preds)
    return f1, cm, total_loss/len(loader)


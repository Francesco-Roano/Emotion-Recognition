import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os, pickle
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import copy


def reconstruction_loss(audio_rec,audio_true):
    return F.mse_loss(audio_rec,audio_true)

def embedding_similarity_loss(x,x_rec, mode="mse+cos", cos_weight=0.1,return_singles=False):
    
    if mode=="ce":
        return F.cross_entropy(x_rec,x)

    #cos_sim = F.cosine_similarity(x, x_rec, dim=-1)  # shape (B, *,)
    #mean_cos_sim = cos_sim.mean() 
    mean_cos_sim = 0

    if mode == "mse":
        loss = F.mse_loss(x, x_rec)
    elif mode == "cos":
        loss = 1 - mean_cos_sim
    elif mode == "mse+cos":
        loss_mse = F.mse_loss(x, x_rec)
        loss_cos = 1 - mean_cos_sim
        loss = loss_mse + cos_weight * loss_cos
    else:
        raise ValueError(f"Invalid mode {mode}, choose 'mse', 'cos', or 'mse+cos'.")
    
    if return_singles:
        return loss,loss_mse,loss_cos
    else:
        return loss

def softF1Loss(probs,targets,eps=1e-8):
    targets_onehot = F.one_hot(targets,num_classes=probs.size(1)).float()
    TP = (probs*targets_onehot).sum(dim=0)
    FP = (probs*(1-targets_onehot)).sum(dim=0)
    FN = ((1-probs)*targets_onehot).sum(dim=0)
    f1 = 2*TP/(2*TP+FP+FN+eps)
    f1_macro = f1.mean()
    loss = 1-f1_macro
    return loss, f1_macro

def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets)

def split_loaders(dataset,batch_size,val_ratio=0.2):
    n=len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(n*(1-val_ratio))
    train_idx = idx[:split]
    test_idx = idx[split:]
    train_ds = Subset(dataset,train_idx)
    test_ds = Subset(dataset,test_idx)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=False)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False)
    return train_dl,test_dl

def train_val_test(dataset,batch_size,val_ratio=0.15,test_ratio=0.15,from_idx=False,file="/home/roano/standalone/split_idx.json",return_dataloader=True):
    if not from_idx:
        n=len(dataset)
        idx = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        split1 = int(n*(1-val_ratio-test_ratio))
        split2 = int(n*(1-test_ratio))
        train_idx = idx[:split1]
        val_idx = idx[split1:split2]
        test_idx = idx[split2:]
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }
        with open(file, "wb") as f:
            pickle.dump(split_indices,f)
    else:
        with open(file, "rb") as f:
            split_indices = pickle.load(f)
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
    train_ds = Subset(dataset,train_idx)
    val_ds = Subset(dataset,val_idx)
    test_ds = Subset(dataset,test_idx)
    train_dl = DataLoader(train_ds,batch_size,shuffle=False)
    val_dl = DataLoader(val_ds,batch_size,shuffle=False)
    test_dl = DataLoader(test_ds,batch_size,shuffle=False)
    if return_dataloader:
        return train_dl,val_dl,test_dl
    else:
        return train_ds,val_ds,test_ds

def make_sample_weights(dataset,class_weights):
    sample_weights = []
    for i in range(len(dataset)):
        label = dataset[i]["label"]
        sample_weights.append(class_weights[label])
    return torch.DoubleTensor(sample_weights)

def make_weighted_loaders(dataset,batch_size,val_ratio=0.15,test_ratio=0.15,from_idx=False):
    train_ds,val_ds,test_ds = speaker_disjoint_split(dataset,batch_size,val_ratio,test_ratio,from_idx=from_idx,return_dataloader=False)
    
    # train [0.9216, 0.9252, 1.0648, 0.8374, 0.3933, 1.8577]
    if from_idx:
        class_weights = torch.load('/home/roano/standalone/class_weights_smart.pt')
    else:
        class_weights = compute_class_weights(train_ds,6)
    train_sample_weights = make_sample_weights(train_ds,class_weights)
    train_sampler = WeightedRandomSampler(train_sample_weights,len(train_sample_weights))
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # val
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    # test
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return train_dl,val_dl,test_dl


def compute_class_weights(dataset, num_classes):
    labels = [dataset[i]["label"].item() for i in range(len(dataset))]
    counts = Counter(labels)

    weights = torch.zeros(num_classes)
    total = sum(counts.values())

    for c in range(num_classes):
        weights[c] = total / (counts[c] + 1e-8)

    weights = weights / weights.mean()
    return weights

class SupConLoss(nn.Module):
    def __init__(self,temperature=0.7,cos_weight=0.2,sup_weight=0.1, return_singles=False):
        super().__init__()
        self.temperature = temperature
        self.cos_weight = cos_weight
        self.sup_weight = sup_weight
        self.return_singles = return_singles

    def forward(self,x,x_rec,z,labels):

        # Supervised Contrastive Loss
        z = F.normalize(z,dim=1)
        B = z.size(0)
        sim = torch.matmul(z,z.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels,labels.T).float().to(z.device)
        logits_mask = 1 - torch.eye(B,device=z.device)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1,keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        supcon_loss = -mean_log_prob_pos.mean()

        # Reconstruction Loss
        rec_loss,mse_loss,cos_loss = embedding_similarity_loss(x,x_rec,cos_weight=self.cos_weight,return_singles=True)
        
        if self.return_singles:
            return rec_loss + self.sup_weight * supcon_loss, mse_loss,cos_loss,supcon_loss
        else:
            return rec_loss + self.sup_weight * supcon_loss

def speaker_disjoint_split(dataset,batch_size,val_ratio=0.15,test_ratio=0.15,from_idx=False,path="/home/roano/standalone/split_idx_speaker_disjoint.json",return_dataloader=True):
    if from_idx:
        with open(path, "rb") as f:
            split_indices = pickle.load(f)
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
    else:
        speaker_to_idx = {}
        for i in range(len(dataset)):
            spk = dataset[i]['speaker']
            speaker_to_idx.setdefault(spk,[]).append(i)
        speakers = list(speaker_to_idx.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(speakers)
        n = len(speakers)
        n_test = int(n*test_ratio)
        n_val = int(n*val_ratio)
        test_speakers = speakers[:n_test]
        val_speakers = speakers[n_test:n_test+n_val]
        train_speakers = speakers[n_test+n_val:]
        
        def collect(speaker_list):
            idx = []
            for s in speaker_list:
                idx.extend(speaker_to_idx[s])
            return idx

        train_idx = collect(train_speakers)
        val_idx = collect(val_speakers)
        test_idx = collect(test_speakers)
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }
        with open(path, "wb") as f:
            pickle.dump(split_indices,f)
    
    train_ds = copy.deepcopy(dataset)
    val_ds = copy.deepcopy(dataset)
    test_ds = copy.deepcopy(dataset)
    train_ds.mode = 'train' 
    val_ds.mode = 'eval'    
    test_ds.mode = 'eval'
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    test_subset = Subset(test_ds, test_idx)
    
    if return_dataloader:
        return (
            DataLoader(train_subset,batch_size,shuffle=False),
            DataLoader(val_subset,batch_size,shuffle=False),
            DataLoader(test_subset,batch_size,shuffle=False)
        )
    else:
        return (
            train_subset,
            val_subset,
            test_subset
        )
    
class NewLoss(nn.Module):
    def __init__(self,temperature=0.7,cos_weight=0.1,sup_weight=0.1,ce_weight=0.05):
        super().__init__()
        self.supcon = SupConLoss(temperature,cos_weight,sup_weight)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self,x,x_rec,z,labels,logits):
        supcon_loss = self.supcon(x,x_rec,z,labels)
        ce_loss = self.ce(logits,labels)
        return supcon_loss + self.ce_weight * ce_loss
    
class AttentionLoss(nn.Module):
    def __init__(self,temperature=0.1,ce_weight=0.2,cons_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.cons_weight = cons_weight
    
    def forward(self,z,z_orig,labels,logits):
        
        z = F.normalize(z,dim=1)
        B = z.size(0)
        sim = torch.matmul(z,z.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels,labels.T).float().to(z.device)
        logits_mask = 1 - torch.eye(B,device=z.device)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1,keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        supcon_loss = -mean_log_prob_pos.mean()

        ce_loss = self.ce(logits,labels.squeeze(1))

        cons_loss = F.mse_loss(z,z_orig)

        return supcon_loss + self.ce_weight * ce_loss + self.cons_weight * cons_loss

def knn_validation(model,train_dl,val_dl,device,k=21):
    model.eval()
    train_feats,train_labels = extract_embeddings(model,train_dl,device)
    test_feats,test_labels = extract_embeddings(model,val_dl,device)
    
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats = scaler.fit_transform(test_feats)

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='cosine',
        weights='distance'
    )

    knn.fit(train_feats,train_labels)
    preds = knn.predict(test_feats)

    return f1_score(test_labels,preds, average='macro')

def extract_embeddings(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch["audio"] = batch["audio"].to(device)
            batch["video"] = batch["video"].to(device)
            za,zv = model(batch)
            z = torch.cat([za,zv],dim=1)
            all_emb.append(z.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    
    return np.concatenate(all_emb), np.concatenate(all_labels)
    
def extract_embeddings_onlyvideo(model,loader,device):
    model.eval()
    all_emb = []
    all_labels = []
    all_speakers = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch["video"] = batch["video"].to(device)
            batch["audio"] = batch["audio"].to(device)
            za,zv = model(batch)
            all_emb.append(zv.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            all_speakers.append(batch['speaker'].numpy())
    return np.concatenate(all_emb), np.concatenate(all_labels), np.concatenate(all_speakers)

def extract_embeddings_onlyaudio(model,loader,device):
    #enc = model.encoder.to(device)
    enc = model
    enc.eval()
    all_emb = []
    all_labels = []
    all_speakers = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch["audio"] = batch["audio"].to(device)
            batch["video"] = batch["video"].to(device)
            za,_ = enc(batch)
            all_emb.append(za.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            all_speakers.append(batch['speaker'].numpy())
    return np.concatenate(all_emb), np.concatenate(all_labels), np.concatenate(all_speakers)

class Loss_3(nn.Module):
    def __init__(self,device,temperature=0.07,sup_weight=1,ce_weight=0.2,margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.sup_weight = sup_weight
        self.ce_weight = ce_weight
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self,z,logits,labels):

        # Supervised Contrastive Loss
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T)
        batch_size = z.shape[0]
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(self.device)
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(self.device), 
            0
        )
        pos_mask = pos_mask * logits_mask
        sim_matrix_margin = sim_matrix - (pos_mask * self.margin)
        logits_sup = sim_matrix_margin / self.temperature
        logits_max, _ = torch.max(logits_sup, dim=1, keepdim=True)
        logits_sup = logits_sup - logits_max.detach()
        exp_logits = torch.exp(logits_sup) * logits_mask 
        log_prob = logits_sup - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-9)
        supcon_loss = -mean_log_prob_pos.mean()

        # Cross Entropy Loss
        ce_loss = self.ce_loss(logits,labels.squeeze(1))

        return self.sup_weight * supcon_loss + self.ce_weight * ce_loss

def extract_embeddings_attention(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch["audio"] = batch["audio"].to(device)
            batch["video"] = batch["video"].to(device)
            z,_ = model(batch)
            all_emb.append(z.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    
    return np.concatenate(all_emb), np.concatenate(all_labels)
        

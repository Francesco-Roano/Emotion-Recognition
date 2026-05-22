from utils import embedding_similarity_loss, softF1Loss, cross_entropy_loss
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

def train_audio_autoencoder(embed,model,loader,optimizer,device="cpu",p=1,loss_mode="mse"):
    model.train()
    total_loss = 0.0
    batch_loss = []
    for i,batch in enumerate(loader):
        #if i/len(loader)>p:
        #    continue
        audio = batch["audio"].to(device)  # (batch,time)
        optimizer.zero_grad()
        if embed is not None:
            audio = embed(audio)
        rec,_ = model(audio)
        loss = embedding_similarity_loss(audio,rec,mode=loss_mode)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*audio.shape[0]
        #if i%10==0:
            #print(i,' batches processed')

    return total_loss/len(loader.dataset)/p,batch_loss

def train_audio_autoencoder_2(model,loader,optimizer,loss_fn,device="cpu"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        audio = batch["audio"].to(device)  # (batch,time)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        rec,z,logits = model(audio)
        loss = loss_fn(audio,rec,z,labels,logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader.dataset)

def train_audio_autoencoder_3(model,loader,optimizer,loss_fn,device="cpu"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        audio = batch["audio"].to(device)  # (batch,time)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        z,logits = model(audio)
        #loss = loss_fn(z,logits,labels)
        loss = loss_fn(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader.dataset)

def train_video_autoencoder(embed,model,loader,optimizer,device="cpu",p=1,loss_mode="mse"):
    model.train()
    total_loss = 0.0
    enc_time = 0.0
    batch_loss = []
    for i, batch in enumerate(loader):
        if i/len(loader)>p:
                continue
        video = batch["video"].to(device)
        if embed is not None:
            video, t = embed(video)
            enc_time += t/video.size(1)
        optimizer.zero_grad()
        rec, _ = model(video)
        loss = embedding_similarity_loss(video,rec,mode=loss_mode)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*video.shape[0]
        #if i%10==0:
        #    print(i," batches processed")
    return total_loss/len(loader.dataset)/p,batch_loss,enc_time/len(loader)

def train_video_autoencoder_2(model,loader,optimizer,loss_fn,device="cpu"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        rec, z, logits = model(video)
        loss = loss_fn(video,rec,z,labels,logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader.dataset)

def train_video_autoencoder_3(model,loader,optimizer,loss_fn,device="cpu"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        z, logits = model(video)
        loss = loss_fn(z,logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader.dataset)

def train_fullmodel(model,loader,optimizer,device,criterion):
    model.train()
    total_loss = 0.0
    batch_loss = []
    l = len(loader)
    for i, batch in enumerate(loader):
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        batch["label"] = batch["label"].to(device)
        if "audio_mask" in batch:
            batch["audio_mask"] = batch["audio_mask"].to(device)
        if "video_mask" in batch:
            batch["video_mask"] = batch["video_mask"].to(device)
            
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(batch)
        loss = criterion(logits,labels)
        batch_loss.append(loss.item())
        loss.backward()

        """
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
        print("grad_norm:", total_norm)
        """

        optimizer.step()
        total_loss += loss.item()
        #print(i," train batches processed")
    return total_loss/l,batch_loss

def overfitting_training(model,loader,optimizer,rep):
    model.train()
    batch = next(iter(loader))
    total_loss = 0
    batch_loss = []
    targets = batch["label"]
    for i in range(rep):
        _, logits = model(batch)
        optimizer.zero_grad()
        loss = cross_entropy_loss(logits,targets)
        loss.backward()
        batch_loss.append(loss.item())
        optimizer.step()
        total_loss += loss.item()
        print(f"{i+1} iterations completed, batch loss is now {loss.item():.6f}")
    return total_loss/rep, batch_loss

def train_cross_attention(model,loader,optimizer,device,loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        z,logits = model(batch)
        loss = loss_fn(z,logits,labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)

def train_mlp_encoder(model,loader,optimizer,device,loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch["audio"] = batch["audio"].to(device)
        batch["video"] = batch["video"].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)

def pretrain_encoder4(model,loader,optimizer,device,loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch['label']
        optimizer.zero_grad()
        za,zv = model(batch)
        loss,_ = loss_fn(za,zv,labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss/len(loader)

def pretrain_encoder4_acc(model,loader,optimizer,device,loss_fn,acc_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i,batch in enumerate(loader):
        batch = {k: v.to(device) for k,v in batch.items()}
        labels = batch['label']
        za,zv = model(batch)
        loss,_ = loss_fn(za,zv,labels)
        total_loss += loss.item()
        loss /= acc_steps
        loss.backward()
        if (i+1)%acc_steps==0:
            optimizer.step()
            optimizer.zero_grad()
    return total_loss/len(loader)

def train_fullmodel4(model,loader,optimizer,device,loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch['label']
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss/len(loader)

def train_fullmodel_aug(model,loader,optimizer,device,loss_fn):
    model.train()
    total_loss = 0.0
    alpha = 0 # mixup intensity
    p_audio_drop = 0.15
    p_video_drop = 0.15
    for batch in loader:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)
        B = audio.size(0)
        if alpha>0:
            lam = np.random.beta(alpha,alpha)
            index = torch.randperm(B).to(device)
            a = lam*audio+(1-lam)*audio[index]
            v = lam*video+(1-lam)*video[index]
            labels_a = labels
            labels_b = labels[index]
        else:
            lam = 1.0
            a = audio
            v = video
            labels_a = labels
            labels_b = labels
        p = torch.rand(1).item()
        if p < p_audio_drop:
            a = torch.zeros_like(a)
        elif p < (p_audio_drop+p_video_drop):
            v = torch.zeros_like(v)
        batch_mixed = {'audio':a,'video':v}
        logits = model(batch_mixed)
        if alpha>0:
            loss = lam*loss_fn(logits,labels_a)+(1-lam)*loss_fn(logits,labels_b)
        else:
            loss = loss_fn(logits,labels_a)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss/len(loader)


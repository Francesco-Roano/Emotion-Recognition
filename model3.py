import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, TimesformerModel
from peft import LoraConfig, get_peft_model
import math
import torchaudio.transforms as T

class ArcFaceClassifier(nn.Module):
    def __init__(self,in_features,out_features,s=30.0,m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features,in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    def forward(self,input,label=None):
        cosine = F.linear(F.normalize(input),F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt((1.0-torch.pow(cosine,2)).clamp(0,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1,label.view(-1,1).long(),1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class WeightedLayerPooling(nn.Module):
    def __init__(self,n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers))

    def forward(self,input):
        # input = layer outputs, list of tensors [B,...,D]
        w = F.softmax(self.weights,dim=0)
        out = 0.0
        for wi, li in zip(w,input):
            out = out + wi * li
        return out
    
class AudioEncoder(nn.Module):
    def __init__(self,latent_dim=512,hidden_size=512,layers=(6,9,12),n_classes=6,dropout=0.1):
        super().__init__()
        try:
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        except OSError:
            print("Can't download microsoft/wavlm-base")
            raise
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.layers = layers
        self.pool = WeightedLayerPooling(len(layers))
        self.proj = nn.Sequential(
            nn.Linear(768,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,latent_dim)
        )
        self.classifier = nn.Linear(latent_dim,n_classes)

    def forward(self,audio):
        # audio (B,T)
        out = self.wavlm(audio,output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states
        pooled_layers = []
        for l in self.layers:
            h = hidden_states[l] # (B,N,D)
            h = h.mean(dim=1)
            pooled_layers.append(h)
        z = self.pool(pooled_layers) # (B,D)
        z = self.proj(z)
        logits  = self.classifier(z)
        return z,logits
    
class AttentiveAudioEncoder(nn.Module):
    def __init__(self,latent_dim=128,layers=(6,9,12),n_classes=6,dropout=0.4):
        super().__init__()
        try:
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        except OSError:
            print("Can't download microsoft/wavlm-base")
            raise
        config = LoraConfig(
            r=8,             
            lora_alpha=16,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"], 
            lora_dropout=0.3,
            bias="none",
            layers_to_transform=[5, 6, 7, 8]
        )
        self.wavlm = get_peft_model(self.wavlm, config)
        self.layers = layers
        self.pool = WeightedLayerPooling(len(layers))
        #self.proj = nn.Linear(768,latent_dim)
        self.classifier = ArcFaceClassifier(in_features=768,out_features=6)
        self.t_aggr = AttentiveTemporalPooling()
        
    def forward(self,audio):
        masking = T.TimeMasking(time_mask_param=80) # Valore alto per WavLM
        audio = masking(audio)
        out = self.wavlm(audio,output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states # each is (B,T,D)
        layers = [hidden_states[l] for l in self.layers]
        z = self.pool(layers) # single (B,T,D)
        z = self.t_aggr(z) # (B,D)
        #z = self.proj(z) # (B,latent_dim)
        logits = self.classifier(z)
        return z, logits

class AttentiveTemporalPooling(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,x):
        # x = (B,T,D)
        weights = self.att(x) # (B,T,1)
        weights = F.softmax(weights,dim=1)
        x = x * weights
        out = torch.sum(x,dim=1) # (B,D)
        return out

class VideoEncoder(nn.Module):
    def __init__(self,latent_dim=512,hidden_dim=512,dropout=0.1,selected_layers=(8,9,10,11),n_classes=6):
        super().__init__()
        self.timesformer = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",output_hidden_states=True)
        for p in self.timesformer.parameters():
            p.requires_grad = False
        self.pool = WeightedLayerPooling(len(selected_layers))
        self.layers = selected_layers
        self.proj = nn.Sequential(
            nn.Linear(768,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,latent_dim)
        )
        self.classifier = nn.Linear(latent_dim,n_classes)

    def forward(self,video):
        # video is (B,T,C,H,W)
        out = self.timesformer(video,output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states # tuple (layer,B,N_tokens,D)
        cls_layers = []
        for l in self.layers:
            h = hidden_states[l]
            cls = h[:,0,:]
            cls_layers.append(cls)
        z = self.pool(cls_layers)
        z = self.proj(z)
        logits = self.classifier(z)
        return z, logits

class AVEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_enc = AudioEncoder()
        self.audio_enc.load_state_dict(torch.load('/home/roano/standalone/models/model3_audio_05_1.pt'))
        for p in self.audio_enc.parameters():
            p.requires_grad = False
        for p in self.audio_enc.proj.parameters():
            p.requires_grad = True
        self.video_enc = VideoEncoder()
        self.video_enc.load_state_dict(torch.load('/home/roano/standalone/models/model3_video_05_1.pt'))
        for p in self.video_enc.parameters():
            p.requires_grad = False
        for p in self.video_enc.proj.parameters():
            p.requires_grad = True

    def forward(self,batch):
        with torch.no_grad():
            za,_ = self.audio_enc(batch['audio'])
            zv,_ = self.video_enc(batch['video'])
        return za, zv
    
class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, za, zv):
        # za, zv: (B, D)
        za = za.unsqueeze(1)  # (B,1,D)
        zv = zv.unsqueeze(1)

        # audio attends video
        out_a, _ = self.attn(za, zv, zv)
        za = self.norm(za + out_a)
        #za = self.norm(out_a)

        # video attends audio
        out_v, _ = self.attn(zv, za, za)
        zv = self.norm(zv + out_v)
        #zv = self.norm(out_v)

        return za.squeeze(1), zv.squeeze(1)

class AttentionEncoder(nn.Module):
    def __init__(self,latent_dim=512,n_classes=6,n_layers=2):
        super().__init__()
        self.encoder = AVEncoder()
        layer = nn.TransformerEncoderLayer(latent_dim,4,batch_first=True)
        self.att_fusion = nn.TransformerEncoder(layer,n_layers)
        self.classifier = nn.Linear(2*latent_dim,n_classes)

    def forward(self,batch):
        za,zv = self.encoder(batch)
        x = torch.stack([za,zv],dim=1) # (B,2,latent_dim)
        z = self.att_fusion(x)
        z = z.reshape(z.size(0),-1) # (B,2*latent_dim)
        logits = self.classifier(z)
        return z,logits
    
class MlpFusion(nn.Module):
    def __init__(self,latent_dim=512,n_classes=6,dropout=0.2):
        super().__init__()
        self.encoder = AVEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(2*latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def forward(self,batch):
        with torch.no_grad():
            za,zv = self.encoder(batch)
        z = torch.cat([za,zv],dim=-1)
        logits = self.classifier(z)
        return logits

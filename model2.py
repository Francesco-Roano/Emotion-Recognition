import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, TimesformerModel
import time

class WavLMEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        #self.wavlm = WavLMModel.from_pretrained(r"C:\Users\franc\wavlm",local_files_only=True)
        try:
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        except OSError:
            print("Can't download microsoft/wavlm-base")
            raise
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()
        self.norm = nn.LayerNorm(768)
        
    def forward(self,x):
        if x.dim()==2:
            x = x.squeeze(1) # (batch,1,time)
        x = x.float()
        with torch.no_grad():
            x = self.wavlm(x)["last_hidden_state"] # (batch,seq_len,768) seq_len~=49 for 1s audio
        x = self.norm(x)
        return x

class ConvAudioEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm([49,768])
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (1,49,768) -> (32,25,384)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64,13,192)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128,7,96)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.flat_dim = 128 * 7 * 96
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)      # (B,128,7,96)
        x = x.view(x.size(0), -1)
        return self.fc(x)        # (B,latent_dim)

class ConvAudioDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_shape=(1,49,768)):
        super().__init__()
        self.out_shape = out_shape
        self.fc = nn.Linear(latent_dim, 128 * 7 * 96)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # (128,14,192)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # (64,28,384)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Upsample(size=(49,768), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 96)
        x = self.decoder(x)
        return x.squeeze(1)
    
class AudioAutoencoder(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,audio):
        z = self.encoder(audio)
        rec = self.decoder(z)
        return rec,z

class SupAudioAutoencoder(nn.Module):
    def __init__(self,encoder,decoder,latent_dim,n_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = nn.Linear(latent_dim,n_classes)

    def forward(self,audio):
        z = self.encoder(audio)
        rec = self.decoder(z)
        logits = self.classifier(z)
        return rec,z,logits

class TimeSformerEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",output_hidden_states=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    
    def forward(self,video):
        # (B,T,C,224,224)
        start = time.perf_counter()
        with torch.no_grad():
            x = self.model(video)
        end = time.perf_counter()
        #print("TimeSformer embed took ",(end-start)*1000," ms")
        x = x.last_hidden_state[:,1:,:] # (B,3136,768) for 16 frames (cls token is removed)
        return x.unsqueeze(1),(end-start)/x.size(0) # (B,1,3136,768)

class TimeSformerCNNEncoder(nn.Module):
    def __init__(self, latent_dim=512, D=768):
        super().__init__()

        self.D = D        # embedding dimension
        self.H = 56
        self.W = 56

        # conv stack
        self.encoder = nn.Sequential(
            nn.Conv2d(D, 256, kernel_size=3, stride=2, padding=1),   # 56→28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), # 28→14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 14→7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        # x = (B,1,3136,768)
        B = x.size(0)
        x = x.squeeze(1)            # (B,3136,768)
        x = x.view(B, 56, 56, self.D)  # (B,56,56,768)
        x = x.permute(0,3,1,2)      # (B,768,56,56)

        x = self.encoder(x)         # (B,128,7,7)
        x = torch.flatten(x, 1)     # (B,128*7*7)
        z = self.fc(x)              # (B,latent_dim)
        return z

class TimeSformerCNNDecoder(nn.Module):
    def __init__(self, latent_dim=512, D=768):
        super().__init__()

        self.D = D
        self.H = 56
        self.W = 56

        self.fc = nn.Linear(latent_dim, 128*7*7)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 7→14
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14→28
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28→56
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, self.D, kernel_size=1),  # back to 768 channels
        )

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)              # (B,128*7*7)
        x = x.view(B,128,7,7)       # (B,128,7,7)
        x = self.decoder(x)         # (B,768,56,56)

        x = x.permute(0,2,3,1)      # (B,56,56,768)
        x = x.reshape(B,1,3136,768) # EXACT original shape
        x = x.squeeze(1)
        return x

class VideoAutoencoder(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,video):
        z = self.encoder(video)
        video_rec = self.decoder(z)
        return video_rec,z

class SupVideoAutoencoder(nn.Module):
    def __init__(self,encoder,decoder,latent_dim,n_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self,video):
        z = self.encoder(video)
        video_rec = self.decoder(z)
        logits = self.classifier(z)
        return video_rec,z,logits

class EncodingBlock(nn.Module):
    def __init__(self,latent_dim=512,audio_path="/home/roano/standalone/models/audio_autoencoder_newloss.pt",video_path="/home/roano/standalone/models/video_autoencoder_newloss.pt"):
        super().__init__()
    
        aenc = ConvAudioEncoder(latent_dim=latent_dim)
        adec = ConvAudioDecoder(latent_dim=latent_dim)
        audio_ae = AudioAutoencoder(aenc,adec)
        audio_ae.load_state_dict(torch.load(audio_path))
        self.aenc = audio_ae.encoder
        for p in self.aenc.parameters():
            p.requires_grad = False

        venc = TimeSformerCNNEncoder(latent_dim=latent_dim)
        vdec = TimeSformerCNNDecoder(latent_dim=latent_dim)
        video_ae = VideoAutoencoder(venc,vdec)
        video_ae.load_state_dict(torch.load(video_path))
        self.venc = video_ae.encoder
        for p in self.venc.parameters():
            p.requires_grad = False

        aenc.eval()
        venc.eval()

    def forward(self,batch):
        audio = batch["audio"]
        video = batch["video"]
        with torch.no_grad():
            za = self.aenc(audio)
            zv = self.venc(video)
        return za,zv
    
class SupEncodingBlock(nn.Module):
    def __init__(self,latent_dim=512,n_classes=6):
        super().__init__()
    
        aenc = ConvAudioEncoder(latent_dim=latent_dim)
        adec = ConvAudioDecoder(latent_dim=latent_dim)
        audio_ae = SupAudioAutoencoder(aenc,adec,latent_dim,n_classes)
        audio_ae.load_state_dict(torch.load("/home/roano/standalone/models/audio_autoencoder_optuna_smart.pt"))
        self.aenc = audio_ae.encoder
        for p in self.aenc.parameters():
            p.requires_grad = False

        venc = TimeSformerCNNEncoder(latent_dim=latent_dim)
        vdec = TimeSformerCNNDecoder(latent_dim=latent_dim)
        video_ae = SupVideoAutoencoder(venc,vdec,latent_dim,n_classes)
        video_ae.load_state_dict(torch.load("/home/roano/standalone/models/video_autoencoder_optuna_smart.pt"))
        self.venc = video_ae.encoder
        for p in self.venc.parameters():
            p.requires_grad = False

        aenc.eval()
        venc.eval()

    def forward(self,batch):
        audio = batch["audio"]
        video = batch["video"]
        with torch.no_grad():
            za = self.aenc(audio)
            zv = self.venc(video)
        return za,zv
    
class EncodingBlockNoCompression(nn.Module):
    def __init__(self,latent_dim=512):
        super().__init__()
        self.afc = nn.Linear(768,latent_dim)
        self.vfc = nn.Linear(768,latent_dim)

    def forward(self,batch):
        za = batch["audio"].mean(dim=1)
        zv = batch["video"].mean(dim=1)
        za = self.afc(za)
        zv = self.vfc(zv)
        return za,zv

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



class Fusion(nn.Module):
    def __init__(self,latent_dim,out_dim,gating=None):
        super().__init__()
        if 2*latent_dim != out_dim:
            self.fc = nn.Sequential(
                nn.Linear(latent_dim*2,out_dim),
                nn.GELU()
            )
        else:
            self.fc = None
        self.gating = gating

    def forward(self,za,zv):
        if self.gating is not None:
            za,zv = self.gating(za,zv)
        fused = torch.cat([za,zv],dim=-1)
        if self.fc is not None:
            fused = self.fc(fused)
        return fused
    
class SupFusion(nn.Module):
    def __init__(self,latent_dim,n_classes,gating=None):
        super().__init__()
        self.gating = gating
        self.head = nn.Linear(2*latent_dim,n_classes)
    
    def forward(self,za,zv):
        if self.gating is not None:
            za,zv = self.gating(za,zv)
        fused = torch.cat([za,zv],dim=-1)
        logits = self.head(fused)
        return fused,logits
    
class MlpClassifier(nn.Module):
    def __init__(self,n_classes,in_dim=512,dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(in_dim // 4, n_classes)
        )
    
    def forward(self,x):
        logits = self.net(x)
        probs = F.softmax(logits,dim=1)
        return probs, logits

class SmallClassifier(nn.Module):
    def __init__(self,n_classes,in_dim,dropout=0.4):
        super().__init__()
        self.net = nn.Linear(in_dim,n_classes)
    
    def forward(self,x):
        logits = self.net(x)
        probs = F.softmax(logits,dim=1)
        return probs, logits

class FullModel(nn.Module):
    def __init__(self,encoding,fusion,classifier):
        super().__init__()
        self.encoding = encoding
        self.fusion = fusion
        self.classifier = classifier

    def forward(self,batch):
        with torch.no_grad():
            za,zv = self.encoding(batch)
        fused = self.fusion(za,zv)
        self.fused = fused
        probs, logits = self.classifier(fused)
        return probs, logits
    
class AttentionEncoder(nn.Module):
    def __init__(self,latent_dim,n_classes,from_pretrained=False,path="/home/roano/standalone/models/cross_attention.pt"):
        super().__init__()
        self.encoder = SupEncodingBlock(latent_dim,n_classes)
        att = CrossAttention(512)
        self.fusion = SupFusion(512,6,att)
        if from_pretrained:
            self.fusion.load_state_dict(torch.load(path))

    def forward(self,batch):
        with torch.no_grad():
            za,zv = self.encoder(batch)
        z,logits = self.fusion(za,zv)
        return z,logits,torch.cat([za,zv],dim=-1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from einops import rearrange
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model
from model3 import WeightedLayerPooling, CrossAttention, AttentiveTemporalPooling
from dan import DAN
from model4 import RobustCrossAttentionFusion


class DepthwiseSeparableTCNBlock(nn.Module):
    def __init__(self,channels,kernel_size=3,dilation=1):
        super().__init__()
        padding = (kernel_size-1)*dilation
        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels
        )
        self.pointwise = nn.Conv1d(channels,channels,kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
    def forward(self,x):
        # x: (B,C,T)
        out = self.depthwise(x)
        out = out[:,:,:-self.depthwise.padding[0]] # causal trim
        out = self.pointwise(out)
        out = self.norm(out)
        return F.relu(out + x)
    
class LiteTCN(nn.Module):
    def __init__(self,channels,num_layers=2):
        super().__init__()
        dilations = [2**i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels,kernel_size=3,dilation=d
            ) for d in dilations
        ])
    def forward(self,x):
        # x: (B,T,C)
        x = x.transpose(1,2) # (B,C,T)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1,2) # (B,T,C)
    
class MobileNetV3VideoEncoder(nn.Module):
    def __init__(self, out_dim=256, tcn_layers=4, dropout = 0.5):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        self.backbone = model.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone_out = 576
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.backbone_out, out_dim)
        self.tcn = LiteTCN(out_dim,num_layers=tcn_layers)
        #self.tpool = AttentiveTemporalPooling(out_dim)
        #self.tpool = StdPooling(out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,video):
        # video: (B,T,C,H,W)
        B = video.shape[0]
        video = rearrange(video,'b t c h w -> (b t) c h w') # (B*T,C,H,W)
        with torch.no_grad():
            feats = self.backbone(video)
        feats = self.pool(feats).squeeze(-1).squeeze(-1) # (B*T,576)
        feats = self.proj(feats) # (B*T,out_dim)
        feats = rearrange(feats,'(b t) c -> b t c',b=B) # (B,T,out_dim)
        feats = self.dropout(feats)
        feats = self.tcn(feats) # (B,T,D)
        #return self.tpool(feats)
        return feats.mean(dim=1)
    
class WavLMAudioEncoder(nn.Module):
    def __init__(self,out_dim=256,lora_layers=[4,5,6,7],dropout=0.4,n_tcn_layers=4):
        super().__init__()
        
        wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base',use_safetensors=True)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"], 
            lora_dropout=0.3,
            bias="none",
            layers_to_transform=lora_layers
        )
        self.model = get_peft_model(wavlm,config)
        '''
        self.model = WavLMModel.from_pretrained('microsoft/wavlm-base')
        for p in self.model.parameters():
            p.requires_grad_ = False
        '''
        self.pool = WeightedLayerPooling(12)
        self.proj = nn.Sequential(
            nn.Linear(768,out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tcn = LiteTCN(out_dim,n_tcn_layers)
        #self.gru = nn.GRU(input_size=out_dim, hidden_size=out_dim//2, 
                          #num_layers=1, batch_first=True, bidirectional=True)
        self.tpool = AttentiveTemporalPooling(out_dim)
        #self.tpool = StdPooling(256)
    def forward(self,audio):
        # audio: (B,16000)
        
        out = self.model(audio,output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states # each is (B,T,D)
        layers = [hidden_states[l] for l in list(range(13))]
        z = self.pool(layers)
        z = self.proj(z) # (B,49,256)
        z = self.tcn(z) 
        #return self.tpool(z)
        #return z.mean(dim=1)
        out = self.tpool(z)
        #out = torch.cat([h_n[0], h_n[1]], dim=1) 
        return out
    
class MultimodalEncoder(nn.Module):
    def __init__(self,out_dim=256,tcn_layers=2,lora_layers=[4,5,6,7],audio_dropout=0.4):
        super().__init__()
        self.audio_encoder = WavLMAudioEncoder(out_dim,lora_layers,audio_dropout,tcn_layers)
        #self.video_encoder = MobileNetV3VideoEncoder(out_dim,tcn_layers)
        self.video_encoder = DANVideoEncoder(tcn_layers=tcn_layers)
        self.audio_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
    def forward(self,batch):
        audio = batch['audio']
        za = self.audio_encoder(audio)
        za = self.audio_proj(za)
        za = F.normalize(za,p=2,dim=1)
        video = batch['video']
        zv = self.video_encoder(video)
        zv = self.video_proj(zv)
        zv = F.normalize(zv,p=2,dim=1)
        return za, zv
    
class FullModel4(nn.Module):
    def __init__(self,encoder_weights,dim=256,n_classes=6,hidden_dim=128,tcn_layers=2):
        super().__init__()
        encoder = MultimodalEncoder(out_dim=dim,tcn_layers=tcn_layers)
        encoder.load_state_dict(torch.load(encoder_weights))
        self.audio_encoder = encoder.audio_encoder
        self.video_encoder = encoder.video_encoder
        '''
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.video_encoder.parameters():
            p.requires_grad = False
        '''
        #self.attention = CrossAttention(dim)
        self.attention = RobustCrossAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(2*dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,n_classes)
        )
        
        # Real-time inference parameters
        self.ema_alpha = 0.3
        self.confidence_threshold = 0.55
        self.register_buffer('ema_logits', torch.zeros(1, n_classes))
        self.is_first_frame = True
        self.enable_ema = False

    def reset_ema(self):
        self.is_first_frame = True

    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        audio_mask = batch.get('audio_mask', None)
        video_mask = batch.get('video_mask', None)
        
        za = self.audio_encoder(audio)
        zv = self.video_encoder(video)
        z = self.attention(za,zv, audio_mask=audio_mask, video_mask=video_mask)
        #z = torch.cat([za,zv],dim=1)
        logits = self.classifier(z)
        
        if not self.training and self.enable_ema:
            if self.is_first_frame:
                self.ema_logits = logits.detach()
                self.is_first_frame = False
            else:
                self.ema_logits = self.ema_alpha * logits.detach() + (1 - self.ema_alpha) * self.ema_logits
            return self.ema_logits
            
        return logits
    
class StdPooling(nn.Module):
    def __init__(self,input_dim,output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.proj = nn.Linear(2*input_dim,output_dim)
    def forward(self,x):
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1,unbiased=False)
        x = torch.cat((x_mean,x_std),dim=1)
        x = self.proj(x)
        return x
    
class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_z = nn.Linear(dim * 2, dim) 
        
    def forward(self, za, zv):
        concat = torch.cat([za, zv], dim=1)
        z_gate = torch.sigmoid(self.linear_z(concat)) 
        return z_gate * za + (1 - z_gate) * zv

class Fullmodel42(nn.Module):
    def __init__(self,encoder_weights=None,dim=256,n_classes=6,hidden_dim=128,tcn_layers=2,video_backbone='mobilenet',pool_type='mean',dropout=0.45):
        super().__init__()
        encoder = OptunaMultimodalEncoder(dim,tcn_layers,video_backbone,pool_type,dropout)
        if encoder_weights is not None:
            encoder.load_state_dict(torch.load(encoder_weights))
        self.audio_encoder = encoder.audio_encoder
        self.video_encoder = encoder.video_encoder
        self.fusion = RobustCrossAttentionFusion(dim=dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2*dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,n_classes)
        )
        self.missing_audio_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.missing_video_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        # Real-time inference parameters
        self.ema_alpha = 0.3
        self.confidence_threshold = 0.55
        self.register_buffer('ema_logits', torch.zeros(1, n_classes))
        self.is_first_frame = True
        self.enable_ema = False

    def reset_ema(self):
        self.is_first_frame = True

    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        audio_mask = batch.get('audio_mask', None)
        video_mask = batch.get('video_mask', None)
        
        za = self.audio_encoder(audio)
        zv = self.video_encoder(video)

        # -- NUOVA LOGICA TOKEN --
        if self.training:
            # batch.get restituisce None in inferenza/test
            audio_mask = batch.get('audio_mask', None) 
            video_mask = batch.get('video_mask', None)
            
            if audio_mask is not None:
                # Dove la maschera è False (cioè droppato), mettiamo il token
                # espandiamo [1, Dim] a [Batch, Dim]
                za = torch.where(audio_mask.unsqueeze(1), za, self.missing_audio_token.expand_as(za))
            if video_mask is not None:
                zv = torch.where(video_mask.unsqueeze(1), zv, self.missing_video_token.expand_as(zv))
        
        z = self.fusion(za,zv)
        logits = self.classifier(z)
        
        if not self.training and self.enable_ema:
            # Inference optimizations: EMA smoothing
            if self.is_first_frame:
                self.ema_logits = logits.detach()
                self.is_first_frame = False
            else:
                self.ema_logits = self.ema_alpha * logits.detach() + (1 - self.ema_alpha) * self.ema_logits
            
            # Use smoothed logits for confidence gating evaluation
            smoothed_logits = self.ema_logits
            probs = F.softmax(smoothed_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            
            # In a real deployed scenario, if entropy > threshold (e.g. ~1.0 for 6 classes)
            # or max_prob < confidence_threshold (0.55), we might reject.
            # Here we just output the EMA smoothed logits if not training.
            return smoothed_logits
            
        return logits

class DANVideoEncoder(nn.Module):
    def __init__(self, dan_weights_path='standalone/models/rafdb_epoch21_acc0.897_bacc0.8275.pth', out_dim=256, tcn_layers=4, dropout=0.5):
        super().__init__()
        
        # 1. Inizializza DAN (di default usa ResNet18)
        # AffectNet di solito ha 7 o 8 classi a seconda dei pesi che scarichi
        self.backbone = DAN(num_head=4, num_class=7,pretrained=False) 
        
        # 2. Carica i pesi ufficiali
        checkpoint = torch.load(dan_weights_path, map_location='cpu')
        # A volte i checkpoint salvano lo state_dict dentro una chiave 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            self.backbone.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.backbone.load_state_dict(checkpoint, strict=True)
            
        # 3. CONGELA IL BACKBONE (Cruciale per non overfittare e non appesantire)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "ca" in name or "sa" in name or "bn" in name:
                param.requires_grad = True
            
        # DAN restituisce feature di dimensione 512
        self.proj = nn.Linear(512, out_dim)
        
        # TCN e GRU rimangono identiche a come le avevamo impostate
        self.tcn = LiteTCN(out_dim, num_layers=tcn_layers)
        self.tpool = AttentiveTemporalPooling(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video):
        # video: (B, T, C, H, W)
        B = video.shape[0]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        
            # Il forward di DAN restituisce una tupla: (out_logits, features, attention_heads)
            # A noi servono solo le features latenti (B*T, 512)
        _, feats, _ = self.backbone(video) 
        
        feats = feats.mean(dim=[2,3])
        feats = self.proj(feats) 
        feats = rearrange(feats, '(b t) c -> b t c', b=B) 
        feats = self.dropout(feats)
        
        # Passaggio temporale
        feats = self.tcn(feats)
        #_, h_n = self.gru(feats)
        out = self.tpool(feats)
        #out = torch.cat([h_n[0], h_n[1]], dim=1) 
        return self.dropout(out)
'''
class RobustCrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.attn_a2v = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_v2a = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1_a = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim*2)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, za, zv, audio_mask=None, video_mask=None):
        za_seq = za.unsqueeze(1) # Diventa (Batch, 1, Dim)
        zv_seq = zv.unsqueeze(1) # Diventa (Batch, 1, Dim)
        
        # se una modality manca del tutto (tutta la batch), non facciamo failare l'attention
        # in un batch misto, usiamo key_padding_mask se supportato, oppure un masking esplicito.
        # Poiché la sequence length è 1, key_padding_mask=True droppa tutto e causa NaN.
        # Quindi facciamo un bypass manuale basato sul mask:
        
        out_a, _ = self.attn_a2v(query=za_seq, key=zv_seq, value=zv_seq)
        if video_mask is not None:
            # If video is missing, we don't enrich audio with it. out_a becomes 0 for that sample.
            video_valid = video_mask.view(-1, 1, 1).float()
            out_a = out_a * video_valid
            
        za_enriched = self.norm1_a(za_seq + self.dropout(out_a))
        
        out_v, _ = self.attn_v2a(query=zv_seq, key=za_seq, value=za_seq)
        if audio_mask is not None:
            # If audio is missing, we don't enrich video with it
            audio_valid = audio_mask.view(-1, 1, 1).float()
            out_v = out_v * audio_valid
            
        zv_enriched = self.norm1_v(zv_seq + self.dropout(out_v))
        
        fused = torch.cat([za_enriched, zv_enriched], dim=-1)
        ffn_out = self.ffn(fused)
        out = self.norm2(fused + self.dropout(ffn_out))
        return out.squeeze(1)
'''

class MultimodalEncoder2(nn.Module):
    def __init__(self,out_dim=256,tcn_layers=2,lora_layers=[4,5,6,7],audio_dropout=0.4):
        super().__init__()
        self.audio_encoder = WavLMAudioEncoder(out_dim,lora_layers,audio_dropout,tcn_layers)
        self.video_encoder = DANVideoEncoder(tcn_layers=tcn_layers)
        self.audio_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(out_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,out_dim)
        )
    def forward(self,batch):
        audio = batch['audio']
        za = self.audio_encoder(audio)
        za = self.audio_proj(za)
        za = F.normalize(za,p=2,dim=1)
        video = batch['video']
        zv = self.video_encoder(video)
        zv = self.video_proj(zv)
        zv = F.normalize(zv,p=2,dim=1)
        return za, zv
    
class DynamicTemporalPooling(nn.Module):
    def __init__(self,pool_type,dim):
        super().__init__()
        self.pool_type = pool_type
        if pool_type=='att':
            self.pool = AttentiveTemporalPooling(dim)
        elif pool_type=='gru':
            self.pool = nn.GRU(dim,dim//2,1,batch_first=True,bidirectional=True)
    def forward(self,x):
        if self.pool_type=='att':
            return self.pool(x)
        elif self.pool_type=='gru':
            _,h_n = self.pool(x)
            return torch.cat([h_n[0],h_n[1]],dim=1)
        else:
            return x.mean(dim=1)
        
class OptunaMultimodalEncoder(nn.Module):
    def __init__(self,dim,tcn_layers,video_backbone,pool_type,dropout):
        super().__init__()
        if video_backbone=='dan':
            self.video_encoder = DANVideoEncoder(out_dim=dim,tcn_layers=tcn_layers,dropout=dropout)
        else:
            self.video_encoder = MobileNetV3VideoEncoder(dim,tcn_layers,dropout)
        self.audio_encoder = WavLMAudioEncoder(dim,dropout=dropout,n_tcn_layers=tcn_layers)
        self.audio_encoder.tpool = DynamicTemporalPooling(pool_type,dim)
        if hasattr(self.video_encoder,'tpool'):
            self.video_encoder.tpool = DynamicTemporalPooling(pool_type,dim)
        elif hasattr(self.video_encoder,'pool') and pool_type!='mean':
            pass
        self.audio_proj = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024,dim))
        self.video_proj = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(), nn.Linear(1024,dim))
    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        za = self.audio_encoder(audio)
        za = F.normalize(self.audio_proj(za),p=2,dim=1)
        zv = self.video_encoder(video)
        zv = F.normalize(self.video_proj(zv),p=2,dim=1)
        return za,zv
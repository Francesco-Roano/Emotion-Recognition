import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from einops import rearrange
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model
from model3 import WeightedLayerPooling, CrossAttention, AttentiveTemporalPooling
from dan import DAN


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
        #self.tpool = AttentiveTemporalPooling(out_dim)
        #self.tpool = StdPooling(256)
    def forward(self,audio):
        # audio: (B,16000)
        out = self.model(audio.squeeze(1).squeeze(1),output_hidden_states=True,return_dict=True)
        hidden_states = out.hidden_states # each is (B,T,D)
        layers = [hidden_states[l] for l in list(range(13))]
        z = self.pool(layers)
        z = self.proj(z) # (B,49,256)
        z = self.tcn(z) 
        #return self.tpool(z)
        return z.mean(dim=1)
        #out = self.tpool(z)
        #out = torch.cat([h_n[0], h_n[1]], dim=1) 
        return out
    
class MultimodalEncoder(nn.Module):
    def __init__(self,out_dim=256,tcn_layers=2,lora_layers=[4,5,6,7],audio_dropout=0.4):
        super().__init__()
        self.audio_encoder = WavLMAudioEncoder(out_dim,lora_layers,audio_dropout,tcn_layers)
        self.video_encoder = MobileNetV3VideoEncoder(out_dim,tcn_layers)
        #self.video_encoder = DANVideoEncoder(tcn_layers=tcn_layers)
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
    def __init__(self,encoder_weights=None,dim=256,n_classes=6,hidden_dim=128,tcn_layers=2):
        super().__init__()
        encoder = MultimodalEncoder(out_dim=dim,tcn_layers=tcn_layers)
        if encoder_weights is not None:
            encoder.load_state_dict(torch.load(encoder_weights))
        self.audio_encoder = encoder.audio_encoder
        self.video_encoder = encoder.video_encoder
        '''
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.video_encoder.parameters():
            p.requires_grad = False
        '''
        self.attention = CrossAttention(dim)
        #self.attention = RobustCrossAttentionFusion()
        '''
        self.classifier = nn.Sequential(
            nn.Linear(2*dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,n_classes)
        )
        '''
        self.classifier = nn.Linear(2*dim,n_classes)
    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        za = self.audio_encoder(audio)
        zv = self.video_encoder(video)
        za,zv = self.attention(za,zv)
        z = torch.cat([za,zv],dim=1)
        logits = self.classifier(z)
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
    def __init__(self,encoder_weights=None,dim=256,n_classes=6,hidden_dim=128,tcn_layers=2):
        super().__init__()
        encoder = MultimodalEncoder2()
        if encoder_weights is not None:
            encoder.load_state_dict(torch.load(encoder_weights))
        self.audio_encoder = encoder.audio_encoder
        self.video_encoder = encoder.video_encoder
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.video_encoder.parameters():
            p.requires_grad = False
        self.fusion = RobustCrossAttentionFusion(dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(2*dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,n_classes)
        )
        self.missing_audio_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.missing_video_token = nn.Parameter(torch.randn(1, dim) * 0.02)
    def forward(self,batch):
        audio = batch['audio']
        video = batch['video']
        with torch.no_grad():
            za = self.audio_encoder(audio)
            zv = self.video_encoder(video)
        
        # -- NUOVA LOGICA TOKEN --
        #if self.training:
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
    def forward(self, za, zv):
        za_seq = za.unsqueeze(1) # Diventa (Batch, 1, Dim)
        zv_seq = zv.unsqueeze(1) # Diventa (Batch, 1, Dim)
        out_a, _ = self.attn_a2v(query=za_seq, key=zv_seq, value=zv_seq)
        za_enriched = self.norm1_a(za_seq + self.dropout(out_a))
        out_v, _ = self.attn_v2a(query=zv_seq, key=za_seq, value=za_seq)
        zv_enriched = self.norm1_v(zv_seq + self.dropout(out_v))
        fused = torch.cat([za_enriched, zv_enriched], dim=-1)
        ffn_out = self.ffn(fused)
        out = self.norm2(fused + self.dropout(ffn_out))
        return out.squeeze(1)


class MultimodalEncoder2(nn.Module):
    def __init__(self,out_dim=256,tcn_layers=2,lora_layers=[4,5,6,7,8,9,10,11,12],audio_dropout=0.4):
        super().__init__()
        self.audio_encoder = WavLMAudioEncoder(out_dim,lora_layers,audio_dropout,tcn_layers)
        #self.video_encoder = DANVideoEncoder3(tcn_layers=tcn_layers)
        self.video_encoder = LandmarkGRUEncoder(dropout=audio_dropout,out_dim=out_dim,num_layers=tcn_layers)
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
        #self.missing_audio_token = nn.Parameter(torch.randn(1, out_dim))
        #self.missing_video_token = nn.Parameter(torch.randn(1, out_dim))
    def forward(self,batch):
        #audio_mask = batch.get('audio_mask', torch.ones(batch['audio'].size(0), dtype=torch.bool, device=batch['audio'].device))
        #video_mask = batch.get('video_mask', torch.ones(batch['video'].size(0), dtype=torch.bool, device=batch['video'].device))
        audio = batch['audio']
        za = self.audio_encoder(audio)
        #za = torch.where(audio_mask.unsqueeze(1).to(za.device), za, self.missing_audio_token.to(za.device).expand_as(za))
        za = self.audio_proj(za)
        za = F.normalize(za,p=2,dim=1)
        video = batch['video']
        video_mask = batch.get('video_mask', None)
        zv = self.video_encoder(video, video_mask=video_mask)
        #zv = torch.where(video_mask.unsqueeze(1).to(zv.device), zv, self.missing_video_token.to(zv.device).expand_as(zv))
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
    
class DANVideoEncoder2(nn.Module):
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
        
        # DAN restituisce feature di dimensione 512
        self.proj = nn.Linear(512, out_dim)
        
        # TCN e GRU rimangono identiche a come le avevamo impostate
        self.tcn = LiteTCN(out_dim, num_layers=tcn_layers)
        #self.tpool = AttentiveTemporalPooling(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(512)

    def forward(self, video):
        # video: (B, T, C, H, W)
        self.backbone.eval() 
        B = video.shape[0]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        
            # Il forward di DAN restituisce una tupla: (out_logits, features, attention_heads)
            # A noi servono solo le features latenti (B*T, 512)
        logits,_,heads = self.backbone(video) 
        feats = heads.sum(dim=1)
        feats = self.norm(feats)
        feats = self.proj(feats) 
        feats = rearrange(feats, '(b t) c -> b t c', b=B) 
        feats = self.dropout(feats)
        
        # Passaggio temporale
        feats = self.tcn(feats)
        #out = self.tpool(feats)
        out = feats.mean(dim=1)
        return self.dropout(out)

class DANVideoEncoder3(nn.Module):
    def __init__(self, out_dim=256, tcn_layers=2, dropout_rate=0.4,unfreeze_heads=True):
        """
        Args:
            out_dim: The final output dimension before the parent class projection (default: 256).
            tcn_layers: Number of recurrent layers for temporal aggregation.
            dropout_rate: Dropout applied to spatial features to prevent memorization.
        """
        super().__init__()
        self.unfreeze_heads = unfreeze_heads
        # 1. Spatial Backbone (DAN)
        self.dan = DAN(num_head=4, num_class=7,pretrained=False)

        self.dan.load_state_dict(torch.load('/home/roano/standalone/models/rafdb_epoch21_acc0.897_bacc0.8275.pth')['model_state_dict'], strict=True)
        # We strip the classification head and batch norm
        self.dan.fc = nn.Identity() 
        self.dan.bn = nn.Identity()
        
        # FREEZE DAN initially to prevent spatial overfitting
        for param in self.dan.features.parameters():
            param.requires_grad = False
        if not self.unfreeze_heads:
            for i in range(self.dan.num_head):
                head = getattr(self.dan, f"cat_head{i}")
                for param in head.parameters():
                    param.requires_grad = False
            
        # 2. Temporal Aggregator
        # We use a BiGRU. To achieve `out_dim` (256), the hidden size needs to be out_dim // 2 (128)
        # because the bidirectional outputs are concatenated.
        hidden_size = out_dim // 2 
        
        self.temporal_dropout = nn.Dropout(dropout_rate)
        
        self.temporal_aggregator = nn.GRU(
            input_size=512,         # DAN's output feature size
            hidden_size=hidden_size, 
            num_layers=tcn_layers,  # Using the parameter from MultimodalEncoder2
            batch_first=True, 
            bidirectional=True
        )

    def train(self, mode=True):
        """
        CRITICAL FIX: Override the default train() method.
        Even if requires_grad=False, calling model.train() forces Batch Norm layers 
        to update their running statistics. This destroys the pretrained weights 
        when facing a severe domain shift (like going from RAF-DB to Crema-D).
        """
        super().train(mode)
        if mode:
            # Force the frozen ResNet backbone back into eval() mode
            self.dan.features.eval()
            
            # If attention heads are also frozen, force them to eval() too
            if not self.unfreeze_heads:
                for i in range(self.dan.num_head):
                    getattr(self.dan, f"cat_head{i}").eval()

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.size()
        
        # Reshape to push through the 2D spatial backbone
        x = x.view(B * T, C, H, W)
        
        # Extract features using DAN
        _, _, heads = self.dan(x) 
        spatial_features = heads.sum(dim=1) 
            
        # Reshape back to sequence: (Batch, Time, 512)
        spatial_features = spatial_features.view(B, T, -1)
        
        # Apply dropout to spatial features to force the RNN to learn temporal cues
        spatial_features = self.temporal_dropout(spatial_features)
        
        # Temporal Aggregation
        # hidden shape: (2 * tcn_layers, Batch, hidden_size)
        rnn_out, hidden = self.temporal_aggregator(spatial_features)
        
        # Extract the final hidden state from the LAST layer in both directions
        # hidden[-2] = last layer forward, hidden[-1] = last layer backward
        video_embedding = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
        
        # Returns exactly (Batch, out_dim) to feed into self.video_proj
        return video_embedding


class LandmarkGRUEncoder(nn.Module):
    def __init__(self, in_features=1404, hidden_dim=128, out_dim=256, num_layers=2, dropout=0.4):
        super().__init__()
        
        # 1. Spatial Geometry Extractor (Frame-level)
        # Maps the raw 1404 coordinates into a richer semantic space
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout)           
        )
        # 2. Temporal Aggregator
        # Tracks how the facial geometry changes over the 16 frames
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Final Projection
        # BiGRU doubles the hidden dim, so we project it back down to your parent class's out_dim
        self.proj = nn.Linear(hidden_dim * 2, out_dim)
        
    def forward(self, x, video_mask=None):
        # x shape: (B, 16, 1404)
        B, T, F = x.size()
        
        # Check if any video_mask is False
        #if video_mask is not None and not video_mask.all():
            #print(f"WARNING: video_mask contains False values. False count: {(~video_mask).sum().item()}")
        
        # Flatten to process all frames through the MLP simultaneously
        x_flat = x.view(B * T, F)
        spatial_feats = self.spatial_mlp(x_flat)
        
        # Reshape back to temporal sequence: (B, 16, hidden_dim)
        spatial_feats = spatial_feats.view(B, T, -1)
        
        # Pass through the GRU
        # hidden shape: (num_layers * 2, B, hidden_dim)
        _, hidden = self.gru(spatial_feats)
        
        # Concatenate the final forward and backward states of the last layer
        video_embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Project to the final latent space
        out = self.proj(video_embedding)
        return out
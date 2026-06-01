import torch
import torch.nn as nn
import torch.nn.functional as F

class MERCL_Loss(nn.Module):
    def __init__(self, temp=0.07, lambda_amcl=1.0, lambda_emcl=1.0, lambda_smcl=0.1, alpha=0.9, hnm=False,weighted=False):
        """
        Multimodal Emotion Recognition Contrastive Learning Loss.
        
        Args:
            temp: Temperature parameter for contrastive loss.
            lambda_*: Pesi per i tre termini della loss.
            alpha: Margine per la SMCL (target similarity).
        """
        super().__init__()
        self.temp = temp
        self.lambda_amcl = lambda_amcl
        self.lambda_emcl = lambda_emcl
        self.lambda_smcl = lambda_smcl
        self.alpha = alpha # Target similarity per SMCL (es. 1.0 se cosine sim)
        self.hnm = hnm # hard negative mining
        if weighted:
            self.weights = torch.load('/home/roano/standalone/class_weights_smart.pt')
        self.weighted = weighted

    def forward(self, z_audio, z_video, labels, audio_energy=None, energy_thresh=0.01):
        """
        Input:
            z_audio: (Batch, D) - Embeddings audio proiettati e normalizzati
            z_video: (Batch, D) - Embeddings video proiettati e normalizzati
            labels: (Batch) - Classi emotive (interi)
            audio_energy: (Batch) - Energia media del segnale audio (opzionale per SMCL)
        """
        
        # 1. AMCL: Intra-Modal Contrastive Learning
        # Audio vs Audio (stessa classe vicini, diversa classe lontani)
        loss_amcl_a = self.supervised_contrastive_loss(z_audio, z_audio, labels, self.weighted)
        # Video vs Video
        loss_amcl_v = self.supervised_contrastive_loss(z_video, z_video, labels, self.weighted)
        
        loss_amcl = (loss_amcl_a + loss_amcl_v) / 2

        # 2. EMCL: Inter-Modal Contrastive Learning
        # Avvicina Audio e Video se hanno la STESSA CLASSE (anche di sample diversi)
        loss_emcl_av = self.supervised_contrastive_loss(z_audio, z_video, labels, self.weighted)
        loss_emcl_va = self.supervised_contrastive_loss(z_video, z_audio, labels, self.weighted)
        
        loss_emcl = (loss_emcl_av + loss_emcl_va) / 2

        # 3. SMCL: Sample-wise Multimodal Alignment
        # Avvicina Audio e Video dello STESSO SAMPLE specifico (ignora la classe)
        # "Minimizes modality gap by aligning representations within the same sample"
        loss_smcl = self.sample_wise_alignment_loss(z_audio, z_video, audio_energy, energy_thresh)

        # Totale pesato
        total_loss = (self.lambda_amcl * loss_amcl) + \
                     (self.lambda_emcl * loss_emcl) + \
                     (self.lambda_smcl * loss_smcl)
                     
        return total_loss, {"AMCL": loss_amcl.item(), "EMCL": loss_emcl.item(), "SMCL": loss_smcl.item()}

    def supervised_contrastive_loss(self, z_anchor, z_contrast, labels, weighted):
        """
        Implementazione vettorizzata della Supervised Contrastive Loss.
        Gestisce sia Intra-modal (z_anchor=z_contrast) che Inter-modal.
        """
        batch_size = z_anchor.shape[0]
        device = z_anchor.device

        # Matrice di similarità (Cosine Similarity dato che z sono normalizzati)
        # (B, B)
        sim_matrix = torch.matmul(z_anchor, z_contrast.T) / self.temp

        # HARD NEGATIVE MINING
        if self.hnm:
            IDX_FEAR = 2
            IDX_NEUTRAL = 4
            IDX_SAD = 5
            labels_col = labels.unsqueeze(1)
            labels_row = labels.unsqueeze(0)
            mask_neutral_sad = ((labels_col==IDX_NEUTRAL)&(labels_row==IDX_SAD))|((labels_col==IDX_SAD)&(labels_row==IDX_NEUTRAL))
            mask_neutral_fear = ((labels_col==IDX_NEUTRAL)&(labels_row==IDX_FEAR))|((labels_col==IDX_FEAR)&(labels_row==IDX_NEUTRAL))
            hard_negative_mask = (mask_neutral_fear|mask_neutral_sad)
            penalty_strength = 1.0
            sim_matrix = sim_matrix + hard_negative_mask * penalty_strength

        # Maschera per le etichette: 1 se labels[i] == labels[j], 0 altrimenti
        labels = labels.view(-1, 1)
        mask_labels = torch.eq(labels, labels.T).float().to(device)

        # Se intra-modal, rimuovi la diagonale (se stessi) dai positivi
        if z_anchor is z_contrast:
            mask_self = torch.eye(batch_size, device=device).bool()
            mask_labels = mask_labels.masked_fill(mask_self, 0) # Rimuovi se stessi
            # Per il denominatore, dobbiamo escludere se stessi dalla somma exp
            # Una tecnica comune è sottrarre un valore grande dalla diagonale di sim_matrix
            sim_matrix = sim_matrix.masked_fill(mask_self, -9e15)

        # Calcolo Log-Sum-Exp per il denominatore (tutti i negativi + positivi)
        # exp_sim = torch.exp(sim_matrix) 
        # log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Metodo numericamente più stabile usando log_softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Media delle log-probabilità solo sui campioni positivi
        # mask_labels.sum(1) conta quanti positivi ci sono per ogni ancora
        # Aggiungiamo epsilon per evitare divisioni per zero
        mean_log_prob_pos = (mask_labels * log_prob).sum(1) / (mask_labels.sum(1) + 1e-8)

        # La loss è il negativo della media
        if not weighted:
            loss = -mean_log_prob_pos.mean()
            return loss
        else:
            if not device == self.weights.device:
                self.weights = self.weights.to(device)
            batch_weights = self.weights[labels]
            weighted_loss = (-mean_log_prob_pos*batch_weights).sum()
            return weighted_loss

    def sample_wise_alignment_loss(self, z_a, z_v, audio_energy=None, thresh=0.01):
        """
        SMCL: Avvicina z_a[i] a z_v[i].
        Usa MSE o Cosine Distance. Qui usiamo MSE sulla similarità come da Eq. 6 (interpretata).
        """
        # Calcolo similarità coseno per ogni coppia corrispondente (diagonale)
        # z_a e z_v sono (B, D) normalizzati -> (z_a * z_v).sum(1) è il coseno
        cos_sim = (z_a * z_v).sum(dim=1) # (B,)
        
        # Loss: vogliamo che cos_sim sia vicino a alpha (es. 1.0)
        # Eq 6 paper: || (z^m)^T p - alpha ||^2
        loss_per_sample = (cos_sim - self.alpha).pow(2)
        
        # Mascheramento basato sull'energia (Audio Energy-based Selection)
        if audio_energy is not None:
            # Crea maschera: 1 se energia > soglia, 0 altrimenti
            mask = (audio_energy > thresh).float()
            # Applica maschera: contiamo solo i campioni con audio valido
            loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss_per_sample.mean()
        
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        #self.alpha = torch.tensor([1.0, 1.0, 1.2, 1.0, 0.8, 1.5])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits (senza softmax)
        if self.alpha is not None and not self.alpha.device == inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=0.1)
        pt = torch.exp(-ce_loss) # probabilità che il modello ha assegnato alla classe corretta
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()
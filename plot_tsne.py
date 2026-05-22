from dataset_2 import CremaEncodedDataset,CremaDSmartLoader, ProcessedCremaDLoader
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from model2 import EncodingBlock, Fusion, FullModel, MlpClassifier, TimeSformerCNNDecoder, TimeSformerCNNEncoder, VideoAutoencoder, ConvAudioDecoder, ConvAudioEncoder, AudioAutoencoder,SupEncodingBlock
from model3 import AudioEncoder,VideoEncoder, AttentiveAudioEncoder
from model4 import MultimodalEncoder2, OptunaMultimodalEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from utils import speaker_disjoint_split,extract_embeddings,extract_embeddings_onlyaudio, extract_embeddings_onlyvideo   
import torch.nn as nn             


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #ds = CremaDSmartLoader(duration=3.0)
    ds = ProcessedCremaDLoader(normalize=True,data_dir='standalone/data/cremad_smart_processed_1s')
    loader  = DataLoader(ds, batch_size=1)
    #_,_,loader = speaker_disjoint_split(ds,1,from_idx=True)

    encoder = MultimodalEncoder2().to(device)
    encoder.load_state_dict(torch.load("/home/roano/standalone/models/final_encoder_1s.pt"))
    emb,lab,spk = extract_embeddings_onlyaudio(encoder,loader,device)
    #emb, lab = extract_embeddings(encoder,loader,device)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42
    )

    emb2d = tsne.fit_transform(emb)

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

    plt.figure()
    sns.scatterplot(
        x=emb2d[:,0],
        y=emb2d[:,1],
        hue=[class_names[l] for l in lab],
        #hue = [s for s in spk],
        palette='tab10',
        s=35,
        alpha=0.8
    )
    plt.title('t-SNE of latent space')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/roano/standalone/figures/presentation_figures/tsne_encoder_1s_audio.png")
    plt.close()


if __name__ == '__main__':
    main()
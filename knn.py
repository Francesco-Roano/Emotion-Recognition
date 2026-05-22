import torch
from model3 import AVEncoder, AttentionEncoder
from utils import speaker_disjoint_split, extract_embeddings_attention
from dataset import CremaDSmartLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    encoder = AttentionEncoder().to(device)
    ds = CremaDSmartLoader()
    train_dl, val_dl, _ = speaker_disjoint_split(ds,1,from_idx=True)
    
    train_feats, train_labels = extract_embeddings_attention(encoder,train_dl,device)
    val_feats, val_labels = extract_embeddings_attention(encoder,val_dl,device)

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    val_feats = scaler.fit_transform(val_feats)

    k_values = [1,3,5,7,11,21,31,41]

    best_f1 = 0

    for k in k_values:

        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric='cosine',
            weights='distance'
        )

        knn.fit(train_feats,train_labels)
        preds = knn.predict(val_feats)

        f1 = f1_score(val_labels,preds, average='macro')
        cm = confusion_matrix(val_labels,preds)

        print(f"k = {k:2d} | f1-macro = {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"kNN Confusion Matrix (k={k}) – F1-macro={f1:.3f}")
            plt.tight_layout()
            plt.savefig("/home/roano/standalone/figures/confusion_matrix_knn_model3_05_1_attention.png")
            plt.close()

if __name__ == '__main__':
    main()
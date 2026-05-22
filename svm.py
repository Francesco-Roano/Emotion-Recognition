import torch
from model2 import EncodingBlock, SupEncodingBlock
from model3 import AttentiveAudioEncoder
from utils import train_val_test, make_weighted_loaders, speaker_disjoint_split, extract_embeddings_onlyaudio
from dataset import CremaDSmartLoader
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    encoder = AttentiveAudioEncoder(layers=list(range(13))).to(device)
    encoder.load_state_dict(torch.load("/home/roano/standalone/models/model3_audioatt_05_1_all_layers.pt"))
    ds = CremaDSmartLoader(n_frames=8)
    train_dl, val_dl, _ = speaker_disjoint_split(ds,1,from_idx=True)
    
    train_feats, train_labels, _ = extract_embeddings_onlyaudio(encoder,train_dl,device)
    val_feats, val_labels, _ = extract_embeddings_onlyaudio(encoder,val_dl,device)

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    val_feats = scaler.fit_transform(val_feats)

    C_values = [0.1,0.5,1.0,5.0,10.0]

    best_f1 = 0

    for C in C_values:

        svm = SVC(
            C=C,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced'
        )

        svm.fit(train_feats,train_labels)
        preds = svm.predict(val_feats)

        f1 = f1_score(val_labels,preds, average='macro')
        cm = confusion_matrix(val_labels,preds)

        print(f"C = {C:.2f} | f1-macro = {f1:.3f}")

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
            plt.title(f"SVM Confusion Matrix (C={C}) – F1-macro={f1:.3f}")
            plt.tight_layout()
            plt.savefig("/home/roano/standalone/figures/confusion_matrix_svm_audio_attenc.png")
            plt.close()

if __name__ == '__main__':
    main()
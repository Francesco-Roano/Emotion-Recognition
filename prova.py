import torch
from model4 import Fullmodel42
from dataset_2 import CremaDSmartLoader
from utils import speaker_disjoint_split
from sklearn.metrics import f1_score
from tqdm import tqdm

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fullmodel42(None).to(device)
    model.load_state_dict(torch.load("/home/roano/standalone/models/fullmodel4_3s_newdan_2.pt"))
    ds = CremaDSmartLoader()
    _,_,dl = speaker_disjoint_split(ds,1,from_idx=True)
    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(dl):
        batch = {k: v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            logits = model(batch)
        preds = torch.argmax(logits,dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())
    f1 = f1_score(all_labels,all_preds,average='macro')
    print(f'F1-score: {f1}')

if __name__ == '__main__':
    main()
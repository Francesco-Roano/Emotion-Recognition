import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

def process_dataset_gpu(input_dir, output_dir):
    # 1. Setup Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Esecuzione su: {device.type.upper()}")
    
    # 2. Inizializza MTCNN su GPU
    # select_largest=True evita che rilevi volti sfocati sullo sfondo
    mtcnn = MTCNN(keep_all=False, device=device, select_largest=True)

    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    # 3. Ciclo di elaborazione
    
    for filename in tqdm(video_files, desc="✂️ Cropping video"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            continue

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # VideoWriter scriverà direttamente il file finale muto
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))
        
        last_box = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # MTCNN richiede immagini RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)
            
            if boxes is not None and len(boxes) > 0:
                last_box = boxes[0]
            
            # Applica il ritaglio se abbiamo una bounding box valida
            if last_box is not None:
                h, w, _ = frame.shape
                x1, y1, x2, y2 = last_box
                
                # Aggiunge un 20% di margine per includere mento e capelli
                bw, bh = x2 - x1, y2 - y1
                margin_x, margin_y = bw * 0.2, bh * 0.2
                
                crop_x1 = max(0, int(x1 - margin_x))
                crop_y1 = max(0, int(y1 - margin_y))
                crop_x2 = min(w, int(x2 + margin_x))
                crop_y2 = min(h, int(y2 + margin_y))
                
                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Sicurezza contro crop vuoti al bordo
                if face_crop.size == 0:
                    face_crop = frame
            else:
                face_crop = frame

            # Ridimensionamento esatto per DAN e salvataggio
            face_crop = cv2.resize(face_crop, (224, 224))
            out.write(face_crop)

        cap.release()
        out.release()

if __name__ == "__main__":
    # Sostituisci con i nomi esatti delle tue cartelle
    INPUT_FOLDER = "/home/roano/standalone/crema-d/VideoMp4"
    OUTPUT_FOLDER = "/home/roano/standalone/crema-d/VideoCropped"
    
    process_dataset_gpu(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Elaborazione completata!")
import torch
import cv2
import numpy as np
import time
import mediapipe as mp
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score  # <-- NEW: For evaluation

# Import your custom modules
from dataset_2 import CremaDSmartLoader
from model4 import Fullmodel42
from utils import speaker_disjoint_split


def _sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def run_benchmark_and_evaluate(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_landmarks = 468
    landmark_dims = 3
    per_frame_features = n_landmarks * landmark_dims
    
    # 1. Initialize MediaPipe in Tracking Mode
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, # Essential for <100ms performance
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. Load Model
    model = Fullmodel42(dim=256).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 3. Initialize Loader 
    print("Initializing DataLoader...")
    ds = CremaDSmartLoader(duration=1.0, augmentor=None,n_frames=8)
    n_frames = ds.n_frames
    _, _, dl = speaker_disjoint_split(ds, 1, from_idx=True)
    
    print(f"Starting benchmark on {device}...")

    # --- 4. WARM-UP PHASE ---
    print("Warming up system (initializing GPU and EGL)...")
    with torch.no_grad():
        for _ in range(5):
            d_audio = torch.randn(1, 48000).to(device)
            d_video = torch.randn(1, n_frames, per_frame_features).to(device)
            _ = model({'audio': d_audio, 'video': d_video, 
                       'audio_mask': torch.tensor([True]).to(device), 
                       'video_mask': torch.tensor([True]).to(device)})
    _sync_if_cuda(device)

    # --- TRACKERS ---
    total_latencies = []
    landmark_latencies = []
    tensor_latencies = []
    inference_latencies = []
    y_true = []  # <-- NEW: Ground truth labels
    y_pred = []  # <-- NEW: Model predictions

    # --- 5. THE MAIN PROFILING LOOP ---
    for batch in tqdm(dl, desc="Benchmarking & Evaluating"):
        # video shape from loader: (B, T, 3, H, W)
        video_tensor_raw = batch['video'] 
        audio_tensor = batch['audio'].to(device)
        curr_n_frames = video_tensor_raw.shape[1]
        
        # Extract the true label for this batch
        true_label = batch['label'].item()  # Assuming batch_size=1
        y_true.append(true_label)

        _sync_if_cuda(device)
        total_start = time.perf_counter()

        # A. MediaPipe Tracking
        landmark_start = time.perf_counter()
        landmarks_buffer = np.zeros((curr_n_frames, n_landmarks, landmark_dims), dtype=np.float32)
        vis_mask = []
        frames_to_process = video_tensor_raw.squeeze(0) 

        for i in range(curr_n_frames):
            # 1. Convert Tensor (C, H, W) -> NumPy (H, W, C)
            frame_np = frames_to_process[i].permute(1, 2, 0).cpu().numpy()
            
            # 2. De-normalize
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
                
            frame_np = np.ascontiguousarray(frame_np)

            # 3. Process 
            res = face_mesh.process(frame_np)
            
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                landmarks_buffer[i] = [[lm.x, lm.y, lm.z] for lm in landmarks]
                vis_mask.append(True)
            else:
                vis_mask.append(False)
        landmark_latencies.append((time.perf_counter() - landmark_start) * 1000)

        # B. Vectorized Normalization (GPU)
        _sync_if_cuda(device)
        tensor_start = time.perf_counter()
        coords = torch.from_numpy(landmarks_buffer).to(device)
        coords = coords - coords[:, 1, :].unsqueeze(1)
        eye_dists = torch.norm(coords[:, 33, :] - coords[:, 263, :], dim=1).view(curr_n_frames, 1, 1)
        video_tensor = (coords / (eye_dists + 1e-6)).view(1, curr_n_frames, per_frame_features)
        _sync_if_cuda(device)
        tensor_latencies.append((time.perf_counter() - tensor_start) * 1000)

        # C. Inference
        batch_input = {
            'audio': audio_tensor,
            'video': video_tensor,
            'audio_mask': torch.tensor([True]).to(device),
            'video_mask': torch.tensor([sum(vis_mask)/curr_n_frames > 0.5]).to(device)
        }

        _sync_if_cuda(device)
        inference_start = time.perf_counter()
        with torch.no_grad():
            logits = model(batch_input)
            # <-- NEW: Get the predicted class
            predicted_class = torch.argmax(logits, dim=-1).item()
            y_pred.append(predicted_class)
        _sync_if_cuda(device)
        inference_latencies.append((time.perf_counter() - inference_start) * 1000)

        total_latencies.append((time.perf_counter() - total_start) * 1000)

    # --- 6. RESULTS ---
    avg_total_latency = np.mean(total_latencies)
    std_total_latency = np.std(total_latencies)

    avg_landmark_latency = np.mean(landmark_latencies)
    avg_tensor_latency = np.mean(tensor_latencies)
    avg_inference_latency = np.mean(inference_latencies)
    
    # <-- NEW: Calculate Macro F1 and Accuracy
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*45)
    print(f"LANDMARK EXTRACTION: {avg_landmark_latency:.2f} ms")
    print(f"TENSOR CREATION:     {avg_tensor_latency:.2f} ms")
    print(f"MODEL INFERENCE:     {avg_inference_latency:.2f} ms")
    print(f"TOTAL LATENCY:       {avg_total_latency:.2f} ms")
    print(f"STD DEVIATION:       {std_total_latency:.2f} ms")
    print(f"STEADY-STATE HZ:     {1000/avg_total_latency:.2f} Hz")
    print("-" * 45)
    print(f"ACCURACY:        {accuracy * 100:.2f}%")
    print(f"MACRO F1-SCORE:  {macro_f1 * 100:.2f}%")
    print("="*45)

    face_mesh.close()

    return {
        'landmark_ms': {
            'mean': float(avg_landmark_latency),
            'std': float(np.std(landmark_latencies)),
            'samples': landmark_latencies,
        },
        'tensor_ms': {
            'mean': float(avg_tensor_latency),
            'std': float(np.std(tensor_latencies)),
            'samples': tensor_latencies,
        },
        'inference_ms': {
            'mean': float(avg_inference_latency),
            'std': float(np.std(inference_latencies)),
            'samples': inference_latencies,
        },
        'total_ms': {
            'mean': float(avg_total_latency),
            'std': float(std_total_latency),
            'samples': total_latencies,
        },
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
    }

if __name__ == "__main__":
    WEIGHTS = "/home/roano/standalone/models/final_model_1s.pt"
    run_benchmark_and_evaluate(weights_path=WEIGHTS)

'''
MediaPipe (16 frames): 45.76ms
Vectorized Normalization: 19.70ms
Model Inference: 12.70ms
TOTAL: 78.16ms

PREDICTED EMOTION: Anger
CONFIDENCE: 86.85%
Total Pipeline Latency: 569.05 ms
Inference Frequency: 1.76 Hz

AVERAGE LATENCY: 131.98 ms
STD DEVIATION:   70.82 ms
STEADY-STATE HZ: 7.58 Hz

3s model
LANDMARK EXTRACTION: 115.56 ms
TENSOR CREATION:     0.46 ms
MODEL INFERENCE:     15.78 ms
TOTAL LATENCY:       132.01 ms
STD DEVIATION:       77.21 ms
STEADY-STATE HZ:     7.58 Hz

1s model
LANDMARK EXTRACTION: 46.69 ms
TENSOR CREATION:     0.52 ms
MODEL INFERENCE:     12.44 ms
TOTAL LATENCY:       59.87 ms
STD DEVIATION:       28.72 ms
STEADY-STATE HZ:     16.70 Hz



'''


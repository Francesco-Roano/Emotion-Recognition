# Emotion-Recognition
Multimodal emotion recognition project for real-time assistive robot. Project developed during Lab training @BraIR Lab, SSSA.
In this repository I included all files used during the development of the final model and its integration in a ROS network. Here the main ones are highlighted:

### Model Architecture
* model4.py (class Fullmodel42)

### Training Scripts
* pretrain_model4.py (contrastive-learning pre training)
* train_fullmodel.py (focal-loss fine tuning)

### Database Loading and Preprocessing
* landmark_extraction.py (processes original DB and saves tensor on disk)
* dataset_2.py (ProcessedCremaDLoader class loads data, MultimodalAugmentor augments)

### Visualization & Metrics
* plot_tsne.py
* inference_time.py

### Useful functions
* train.py
* test.py
* utils.py

### ROS nodes
* camera_node.py
* micro_node.py
* emo_node_2.py 
* emo_2_launch.py (to launch all 3 nodes at once)
* requirements.txt (indicates pip dependencies needed to run the node)

## Emotion Node Instructions
This ROS 2 node performs real-time, multimodal (audio and video) emotion recognition. It utilizes MediaPipe for face mesh extraction and a custom PyTorch neural network to infer the current emotional state. Designed for flexible integration within a larger robotic architecture, it allows you to dynamically toggle sensor modalities and adjust confidence thresholds on the fly via the ROS 2 Parameter Server.

### 📥 Subscribed Topics
* **Video:** (Configurable) Expects `sensor_msgs/msg/Image` (Standard BGR8).
* **Audio:** (Configurable) Expects `std_msgs/msg/Float32MultiArray` (16kHz audio chunks).

### 📤 Published Topics
* `/robot/detected_emotion` (`std_msgs/msg/String`): Outputs the predicted emotion, confidence score, and inference timing metrics.

---

### ⚙️ ROS 2 Parameters

The node relies entirely on the parameter server for configuration. You **must** provide a valid `weights_path` for the node to initialize successfully.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `weights_path` | `string` | `""` | **[Required]** Absolute path to the `.pt` PyTorch weights file. |
| `model_size` | `string` | `'small'` | Selects the buffer window. Options: `'small'` (1s window) or `'large'` (3s window). |
| `video_topic` | `string` | `'/camera/color/image_raw'` | The image topic to subscribe to. |
| `audio_topic` | `string` | `'/microphone/audio_data'` | The audio topic to subscribe to. |
| `enable_video` | `bool` | `True` | Toggles video processing and MediaPipe inference. |
| `enable_audio` | `bool` | `True` | Toggles audio processing. |
| `audio_energy_threshold`| `double` | `0.01` | Minimum RMS energy required to process audio. Below this, audio is ignored. |
| `face_visibility_threshold`| `double`| `0.5` | Minimum ratio of frames in the window where a face must be detected to process video. |
| `confidence_threshold` | `double` | `0.5` | Minimum probability required to publish a specific emotion. Below this, outputs `Neutral`. |
| `record` | `bool` | `False` | Enables recording of the session. Fuses audio and video into an MP4 upon shutdown. |
| `output_dir` | `string` | `'/tmp'` | Directory where the recorded `fused_demo.mp4` and temporary files are saved. |




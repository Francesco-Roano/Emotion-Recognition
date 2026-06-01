# Emotion-Recognition
Multimodal emotion recognition project for real-time assistive robot. Project developed during Lab training @BraIR Lab, SSSA.
In this repository I included all files used during the development of the final model and its integration in a ROS network. Here the main ones are described:

# Model Architecture
-model4.py (class Fullmodel42)

# Training Scripts
-pretrain_model4.py (contrastive-learning pre training)
-train_fullmodel.py (focal-loss fine tuning)

# Database Loading and Preprocessing
-landmark_extraction.py (processes original DB and saves tensor on disk)
-dataset_2.py (ProcessedCremaDLoader class loads data, MultimodalAugmentor augments)

# Visualization & Metrics
-plot_tsne.py
-inference_time.py

# Useful functions
-train.py
-test.py
-utils.py

# ROS nodes
-camera_node.py
-micro_node.py
-emotion_node.py (uses model.py for final model architecture)

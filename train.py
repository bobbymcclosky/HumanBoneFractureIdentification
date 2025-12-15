"""
Human Bone Fracture Identification - Training Script

This module trains a YOLOv11 object detection model to identify bone fractures 
in medical X-ray and MRI images. The training uses transfer learning from a 
pre-trained YOLO model with data augmentation and early stopping.

Author: Bobby McClosky
Dataset: Human Bone Fractures Multi-modal Image Dataset (HBFMID)
Model: YOLOv11 Nano (yolo11n)
"""

from comet_ml import Experiment
from ultralytics import YOLO

# Initialize Comet ML experiment for tracking training metrics and visualizations
# Comet ML provides real-time monitoring of training progress, loss curves, and model performance
experiment = Experiment(
    api_key="UXWTiPNthHanNObk56kiYP6Kh",  # API key for Comet ML authentication
    project_name="general",                # Project name in Comet ML workspace
    workspace="bobbymcclosky",             # Comet ML workspace identifier
)

# Load pre-trained YOLOv11 nano model as the base for transfer learning
# The nano variant provides a good balance between speed and accuracy for medical imaging
model = YOLO('yolo11n.pt')

# Train the model with optimized hyperparameters for bone fracture detection
# Transfer learning approach: freeze early layers and fine-tune later layers
results = model.train(
    # Dataset configuration
    data='Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection/data.yaml',
    
    # Training hyperparameters
    epochs=100,          # Extended training for better convergence on medical imaging data
    imgsz=640,           # Input image size (640x640 pixels) - standard for YOLO
    batch=8,             # Batch size - adjust based on available GPU memory
    device=0,            # GPU device ID (0 for first GPU, use 'cpu' for CPU training)
    
    # Model naming and checkpointing
    name='fracture_best_model',  # Name for this training run (saved in runs/detect/)
    
    # Learning rate schedule
    lr0=.001,            # Initial learning rate - lower for fine-tuning pre-trained model
    lrf=.0001,           # Final learning rate (cosine annealing schedule)
    
    # Transfer learning configuration
    freeze=10,           # Freeze first 10 layers to preserve pre-trained features
    
    # Early stopping and regularization
    patience=15,         # Stop training if validation metric doesn't improve for 15 epochs
    dropout=.5,          # Dropout rate to prevent overfitting (50%)
    
    # Data augmentation techniques
    mosaic=.5,           # Mosaic augmentation probability (combines 4 images)
    mixup=.1,            # MixUp augmentation probability (blends 2 images)
    
    # Output verbosity
    verbose=True         # Print detailed training progress
)

# Validate the trained model on the validation set
# Returns metrics including mAP (mean Average Precision), precision, and recall
metrics = model.val()

print("✓ Test training and evaluation complete!")

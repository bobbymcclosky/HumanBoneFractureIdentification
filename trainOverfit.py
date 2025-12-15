"""
Human Bone Fracture Identification - Overfitting Check Script

This module performs an overfitting test to verify that the model can learn from
a small subset of data. This is a crucial diagnostic step to ensure the model 
architecture and training pipeline are working correctly before full-scale training.

The script intentionally disables regularization and data augmentation to allow
the model to memorize the small training set. If the model can't overfit this
small dataset, it indicates a fundamental problem with the model or training setup.

Author: Bobby McClosky
Purpose: Model architecture and training pipeline validation
"""

from comet_ml import Experiment
from ultralytics import YOLO

# Initialize Comet ML experiment for tracking overfitting check metrics
experiment = Experiment(
    api_key="UXWTiPNthHanNObk56kiYP6Kh",  # API key for Comet ML authentication
    project_name="general",                # Project name in Comet ML workspace
    workspace="bobbymcclosky",             # Comet ML workspace identifier
)

# Load pre-trained YOLOv11 nano model
model = YOLO('yolo11n.pt')

# Train on small dataset with minimal regularization to check if model can overfit
# Success criterion: Training loss should decrease to near zero
results = model.train(
    # Small dataset configuration (typically 50-100 images)
    data='Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone_Fractures_Overfit_Check/data_small.yaml',
    
    # Training hyperparameters
    epochs=50,           # Moderate number of epochs to allow overfitting
    imgsz=640,           # Input image size (640x640 pixels)
    batch=8,             # Batch size
    device=0,            # GPU device ID
    
    # Run identification
    name='overfit_check',  # Name for this diagnostic run
    
    # Learning rate (lower to ensure stable convergence)
    lr0=1e-4,            # Initial learning rate
    lrf=1e-5,            # Final learning rate
    
    # Transfer learning disabled for overfitting check
    freeze=0,            # Don't freeze any layers - allow full model training
    
    # Regularization disabled to encourage overfitting
    patience=0,          # No early stopping - train for all epochs
    dropout=0,           # No dropout regularization
    
    # Data augmentation disabled to allow memorization
    mosaic=False,        # No mosaic augmentation
    mixup=False,         # No mixup augmentation
    
    # Output verbosity
    verbose=True         # Print detailed training progress
)

# Validate the model (should show high performance on this small set if overfitting works)
metrics = model.val()

print("✓ Test training and evaluation complete!")

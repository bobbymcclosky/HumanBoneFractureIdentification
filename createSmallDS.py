"""
Human Bone Fracture Identification - Small Dataset Creator

This module creates a small subset of the training dataset for overfitting checks
and rapid prototyping. The subset includes both training and validation splits
with the same images to enable overfitting verification.

Purpose:
- Create a minimal dataset (50 images) to test model architecture
- Validate training pipeline before full-scale training
- Enable quick iteration during development
- Perform sanity checks on data loading and augmentation

Author: Bobby McClosky
"""

import os
import shutil
import random
from pathlib import Path

# ============================================================================
# Configuration: Source Dataset Paths
# ============================================================================
# Note: These paths are user-specific and should be updated for your system
base_path = Path('/home/byorky/HumanBoneFractureIdentification/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection')
train_images = base_path / 'train' / 'images'  # Source training images
train_labels = base_path / 'train' / 'labels'  # Source YOLO format labels

# ============================================================================
# Configuration: Destination Small Dataset Paths
# ============================================================================
small_subset_path = Path('/home/byorky/HumanBoneFractureIdentification/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone_Fractures_Overfit_Check')
small_train_images = small_subset_path / 'train' / 'images'
small_train_labels = small_subset_path / 'train' / 'labels'
small_valid_images = small_subset_path / 'valid' / 'images'  # Validation uses same images
small_valid_labels = small_subset_path / 'valid' / 'labels'  # for overfitting check

# ============================================================================
# Clean Existing Small Dataset
# ============================================================================
# Remove any existing files from previous runs to ensure fresh dataset
for f in small_train_images.glob('*'):
    f.unlink()
for f in small_train_labels.glob('*'):
    f.unlink()
for f in small_valid_images.glob('*'):
    f.unlink()
for f in small_valid_labels.glob('*'):
    f.unlink()

# ============================================================================
# Sample Images from Training Set
# ============================================================================
# Get all training images (supports multiple image formats)
all_images = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sample a subset of images (50 or fewer if dataset is smaller)
sample_size = min(50, len(all_images))
small_sample = random.sample(all_images, sample_size)

# ============================================================================
# Copy Images and Labels to Small Dataset
# ============================================================================
# Note: Images are copied to both train and valid splits (same data)
# This is intentional for overfitting checks - model should memorize the data
for img_name in small_sample:
    # Copy image to training set
    src_img = train_images / img_name
    dst_train_img = small_train_images / img_name
    dst_valid_img = small_valid_images / img_name
    shutil.copy(src_img, dst_train_img)
    shutil.copy(src_img, dst_valid_img)
    
    # Copy corresponding label file (YOLO format: class x_center y_center width height)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    src_lbl = train_labels / label_name
    dst_train_lbl = small_train_labels / label_name
    dst_valid_lbl = small_valid_labels / label_name
    
    # Only copy label if it exists (some images may not have annotations)
    if src_lbl.exists():
        shutil.copy(src_lbl, dst_train_lbl)
        shutil.copy(src_lbl, dst_valid_lbl)

print(f"✓ Created small subset with {len(small_sample)} images")

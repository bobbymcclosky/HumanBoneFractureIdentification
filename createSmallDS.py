import os
import shutil
import random
from pathlib import Path

# Your dataset paths
base_path = Path('/home/byorky/HumanBoneFractureIdentification/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection')
train_images = base_path / 'train' / 'images'
train_labels = base_path / 'train' / 'labels'

# Create small subset directory
small_subset_path = Path('/home/byorky/HumanBoneFractureIdentification/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone_Fractures_Overfit_Check')
small_train_images = small_subset_path / 'train' / 'images'
small_train_labels = small_subset_path / 'train' / 'labels'
small_valid_images = small_subset_path / 'valid' / 'images'
small_valid_labels = small_subset_path / 'valid' / 'labels'

# Clear old data
for f in small_train_images.glob('*'):
    f.unlink()
for f in small_train_labels.glob('*'):
    f.unlink()
for f in small_valid_images.glob('*'):
    f.unlink()
for f in small_valid_labels.glob('*'):
    f.unlink()

# Get all training images
all_images = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sample 50 images this time
sample_size = min(50, len(all_images))
small_sample = random.sample(all_images, sample_size)

# Copy images and labels
for img_name in small_sample:
    src_img = train_images / img_name
    dst_train_img = small_train_images / img_name
    dst_valid_img = small_valid_images / img_name
    shutil.copy(src_img, dst_train_img)
    shutil.copy(src_img, dst_valid_img)
    
    label_name = os.path.splitext(img_name)[0] + '.txt'
    src_lbl = train_labels / label_name
    dst_train_lbl = small_train_labels / label_name
    dst_valid_lbl = small_valid_labels / label_name
    if src_lbl.exists():
        shutil.copy(src_lbl, dst_train_lbl)
        shutil.copy(src_lbl, dst_valid_lbl)

print(f"✓ Created small subset with {len(small_sample)} images")

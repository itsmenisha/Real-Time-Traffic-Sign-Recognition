import os
import random
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from ultralytics import YOLO
import numpy as np

# ----------------------------
# SETTINGS
# ----------------------------
GTSRB_PATH = Path(
    r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\GTSRB\Train")
OUTPUT_PATH = Path(
    r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\yolo_dataset")
IMG_SIZE = (416, 416)
MAX_IMAGES = 15000
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 200
AUGMENT = True

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ----------------------------
# CREATE YOLO FOLDERS
# ----------------------------
for split in ["train", "val"]:
    (OUTPUT_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

# ----------------------------
# COLLECT ALL IMAGES
# ----------------------------
all_images = []
for class_id in range(43):
    class_folder = GTSRB_PATH / f"{class_id:05d}"
    if not class_folder.exists():
        class_folder = GTSRB_PATH / str(class_id)
    if not class_folder.exists():
        print(f"❌ Class {class_id} missing, skipping")
        continue
    class_images = [p for p in class_folder.glob(
        "*.*") if p.suffix.lower() in [".ppm", ".png", ".jpg", ".jpeg"]]
    for img_path in class_images:
        all_images.append((img_path, class_id))

print(f"Total images found: {len(all_images)}")

# ----------------------------
# SIMPLE RANDOM SELECTION (no heavy feature extraction)
# ----------------------------
selected_images = random.sample(all_images, min(MAX_IMAGES, len(all_images)))
print(f"Selected mini-batch images: {len(selected_images)}")

# ----------------------------
# TRAIN/VAL SPLIT
# ----------------------------
train_images, val_images = train_test_split(
    selected_images,
    test_size=TEST_SIZE,
    stratify=[x[1] for x in selected_images],
    random_state=RANDOM_SEED
)

# ----------------------------
# FUNCTION TO SAVE YOLO LABELS
# ----------------------------


def save_yolo_label(img_path, class_id, img_out_dir, label_out_dir, augment=False):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    img.save(img_out_dir / img_path.name)
    x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
    with open(label_out_dir / f"{img_path.stem}.txt", "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Optional horizontal flip
    if augment:
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_name = img_path.stem + "_flip" + img_path.suffix
        img_flipped.save(img_out_dir / flipped_name)
        with open(label_out_dir / f"{img_path.stem}_flip.txt", "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# ----------------------------
# SAVE TRAIN AND VAL DATA IN BATCHES
# ----------------------------


def process_in_batches(images_list, img_dir, label_dir, augment=False):
    for i in range(0, len(images_list), BATCH_SIZE):
        batch = images_list[i:i+BATCH_SIZE]
        print(
            f"Processing batch {i//BATCH_SIZE + 1} / {(len(images_list)-1)//BATCH_SIZE + 1}")
        for img_path, class_id in batch:
            save_yolo_label(img_path, class_id, img_dir,
                            label_dir, augment=augment)


process_in_batches(train_images, OUTPUT_PATH / "images/train",
                   OUTPUT_PATH / "labels/train", augment=AUGMENT)
process_in_batches(val_images, OUTPUT_PATH / "images/val",
                   OUTPUT_PATH / "labels/val", augment=False)

print(
    f"✅ YOLO dataset ready: {len(train_images)} train, {len(val_images)} val")

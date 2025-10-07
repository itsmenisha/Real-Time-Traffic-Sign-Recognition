import os
import random
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split
from PIL import Image

# ----------------------------
# SETTINGS
# ----------------------------
GTSRB_PATH = Path(
    r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\GTSRB\Train")
COORD_PATH = Path(
    r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\GTSRB\trainvalues")
OUTPUT_PATH = Path(
    r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\mini_dataset")
TARGET_IMAGES = 8000
TEST_SIZE = 0.2
RANDOM_SEED = 42
AUGMENT = True

random.seed(RANDOM_SEED)

# ----------------------------
# CREATE YOLO FOLDERS
# ----------------------------
for split in ["train", "val"]:
    (OUTPUT_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

# ----------------------------
# BUILD MINI DATASET
# ----------------------------
mini_dataset = []

for class_id in range(43):
    class_folder = GTSRB_PATH / str(class_id)
    if not class_folder.exists():
        continue

    class_images = [p for p in class_folder.glob(
        "*.*") if p.suffix.lower() in [".ppm", ".png", ".jpg", ".jpeg"]]

    for i in range(0, len(class_images), 10):
        batch = class_images[i:i+10]
        selected = random.sample(batch, min(2, len(batch)))

        for img_path in selected:
            if len(mini_dataset) >= TARGET_IMAGES:
                break
            txt_file = COORD_PATH / str(class_id) / f"{img_path.stem}.txt"
            if txt_file.exists():
                mini_dataset.append((img_path, txt_file, class_id))

        if len(mini_dataset) >= TARGET_IMAGES:
            break
    if len(mini_dataset) >= TARGET_IMAGES:
        break

print(f"Total mini dataset size: {len(mini_dataset)}")

# ----------------------------
# SPLIT TRAIN/VAL
# ----------------------------
train_data, val_data = train_test_split(
    mini_dataset,
    test_size=TEST_SIZE,
    stratify=[x[2] for x in mini_dataset],
    random_state=RANDOM_SEED
)

# ----------------------------
# FUNCTION TO SAVE IMAGES & LABELS
# ----------------------------


def save_yolo_files(data, split, augment=False):
    img_out_dir = OUTPUT_PATH / "images" / split
    label_out_dir = OUTPUT_PATH / "labels" / split

    for img_path, txt_file, class_id in data:
        # Copy original image
        copy2(img_path, img_out_dir / img_path.name)
        # Copy original label
        copy2(txt_file, label_out_dir / txt_file.name)

        if augment:
            # Augmentation: horizontal flip
            img = Image.open(img_path).convert("RGB")
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_name = img_path.stem + "_flip" + img_path.suffix
            img_flipped.save(img_out_dir / flipped_name)

            # Flip label coordinates horizontally
            with open(txt_file, "r") as f:
                coords = f.read().strip().split()
            if len(coords) == 5:
                cls_id, x_center, y_center, width, height = coords
                x_center = 1.0 - float(x_center)  # flip horizontally
                flipped_label = f"{cls_id} {x_center:.6f} {y_center} {width} {height}\n"
                label_flipped_name = img_path.stem + "_flip.txt"
                with open(label_out_dir / label_flipped_name, "w") as f:
                    f.write(flipped_label)


# ----------------------------
# SAVE TRAIN/VAL
# ----------------------------
print("Saving TRAIN images & labels...")
save_yolo_files(train_data, "train", augment=AUGMENT)

print("Saving VAL images & labels...")
save_yolo_files(val_data, "val", augment=False)


print(f"✅ Mini YOLO dataset ready at {OUTPUT_PATH}")

import os
import random
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# Paths (edit if needed)
GTSRB_ROOT = Path("data/GTSRB/Train")    # Original full training dataset
OUTPUT_ROOT = Path("data/mini_dataset")  # New mini dataset folder
TARGET_SIZE = 10000                      # Total images to keep
IMG_SIZE = (416, 416)                     # Resize to YOLO size


def create_mini_dataset():
    classes = sorted([d.name for d in GTSRB_ROOT.iterdir() if d.is_dir()])
    images, labels = [], []
    for cls in classes:
        for img in (GTSRB_ROOT / cls).glob("*.*"):
            if img.suffix.lower() in [".ppm", ".jpg", ".png"]:
                images.append(img)
                labels.append(cls)

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=TARGET_SIZE, random_state=42
    )

    saved_count = 0  # <-- counter for created images

    for idx, _ in sss.split(images, labels):
        selected = [images[i] for i in idx]
        for img_path in selected:
            cls = img_path.parent.name
            out_dir = OUTPUT_ROOT / "images" / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            img = Image.open(img_path).resize(IMG_SIZE)
            img.save(out_dir / (img_path.stem + ".jpg"))

            saved_count += 1
            if saved_count % 100 == 0:  # print progress every 100 images
                print(f"✔ Saved {saved_count} images so far...")

    print(f"✅ Mini dataset created at {OUTPUT_ROOT}")
    print(f"📸 Total images saved: {saved_count}")


if __name__ == "__main__":
    create_mini_dataset()

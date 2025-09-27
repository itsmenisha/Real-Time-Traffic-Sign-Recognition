import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/train/exp/weights/best.onnx"  # Use ONNX model
IMAGE_DIR = Path("data/mini_dataset/images")     # Test images folder
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_inference():
    model = YOLO(MODEL_PATH)
    images = list(IMAGE_DIR.rglob("*.jpg"))
    for img_path in images[:20]:  # test with first 20 images
        start = time.time()
        results = model.predict(source=str(img_path), device="cpu", save=False)
        elapsed = (time.time() - start) * 1000
        print(f"{img_path.name}: {elapsed:.2f} ms")
        results[0].save(filename=OUTPUT_DIR / f"{img_path.stem}_pred.jpg")


if __name__ == "__main__":
    run_inference()

from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.pt")  # Tiny backbone for speed
    model.train(
        data="mini_dataset.yaml",  # Must point to your mini dataset config
        imgsz=416,
        epochs=50,
        batch=16
    )


if __name__ == "__main__":
    train()

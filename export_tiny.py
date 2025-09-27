from ultralytics import YOLO


def export():
    # Adjust exp path if different
    model = YOLO("runs/train/exp/weights/best.pt")
    model.export(format="onnx", simplify=True)


if __name__ == "__main__":
    export()

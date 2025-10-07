# train_yolo.py
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="mini_dataset.yaml",
        imgsz=416,         
        epochs=50,         
        batch=6,          
        device="cpu",     
        workers=4,         
        cache=True,        
        name="exp_cpu_fast",
        patience=10,       
        augment=True       
    )


if __name__ == "__main__":
    main()

from ultralytics import YOLO

model = YOLO(r"runs/detect/exp_cpu_fast2/weights/best.pt")

results = model.predict(
    source=r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\German-traffic-sign-recognition-benchmark-GTSRB-dataset.png",
    imgsz=5000,      
    conf=0.15,      
    iou=0.45,      
    show=True,       
    save=True,  
    project="runs/detect",
    name="video_test",
    stream=False
)


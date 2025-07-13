from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=20 ,
    imgsz=640,
    batch=8,
    name='detected',
    workers=0,
    device='0',  # or 'cuda' if using GPU
    augment=True,
    lr0=0.001,
    patience=20
)


from ultralytics import YOLO

models = {
    'v5': 'yolov5nu.pt',
    'v8': 'yolov8n.pt',
}

model = YOLO(models['v8'])
model.info()

results = model.train(data='../balloon/data.yaml', epochs=100, imgsz=640)

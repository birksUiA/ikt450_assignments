import os

from ultralytics import YOLO

model = YOLO('./results/yolov8l/weights/last.pt')

test_image_dir = './test_images'
for file in os.listdir(test_image_dir):
    results = model.predict(
        os.path.join(test_image_dir, file),
        save=True,
        augment=True,
        retina_masks=True
    )
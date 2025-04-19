import torch
from ultralytics import YOLO
from config import constants

class YoloContainerDetector:
    def __init__(self):
        self.model = YOLO(constants.YOLO_WEIGHTS_PATH)
        self.model.fuse()

    def detect(self, image):
        results = self.model(image)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2-x1, y2-y1)) 
        return boxes
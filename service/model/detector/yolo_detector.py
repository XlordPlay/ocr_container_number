from ultralytics import YOLO
from service.model.config import constants
import torch

class YoloContainerDetector:
    def __init__(self):
        self.model = YOLO(constants.YOLO_WEIGHTS_PATH, task='detect')
        self.model.fuse()
    
    def detect(self, image):
        results = self.model.predict(
            image, 
            classes=list(constants.TARGET_CLASSES.values()),  # Фильтр по классам
            conf=0.5,  # Порог уверенности
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        boxes_with_classes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = [k for k, v in constants.TARGET_CLASSES.items() if v == class_id][0]
                boxes_with_classes.append((box, class_name))
                
        boxes = []
        for result in results:
            for box in result.boxes:
                if box.cls in constants.TARGET_CLASSES.values():  # Двойная проверка
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1, y1, x2-x1, y2-y1))
        return boxes
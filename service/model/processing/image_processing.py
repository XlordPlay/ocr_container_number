import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def preprocess_for_ocr(image, class_name):  # Добавляем второй аргумент
        """Разная предобработка для разных классов"""
        # Базовый препроцессинг
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if class_name == 'cn-11':
            # Оптимизация для крупного текста
            processed = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        elif class_name in ['cn-7', 'cn-4']:
            # Обработка мелкого текста
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)
            processed = cv2.medianBlur(processed, 3)
            
        return processed
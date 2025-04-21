import cv2
from service.model.detector.yolo_detector import YoloContainerDetector
from service.model.processing.image_processing import ImageProcessor
from service.model.ocr.text_recognizer import OCRProcessor
from service.model.validator.container_validator import ContainerValidator
from service.model.config import constants
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(image_path):
    detector = YoloContainerDetector()
    ocr_processor = OCRProcessor()
    
    image = cv2.imread(image_path)
    boxes = detector.detect(image)
    
    # Сохраняем отладочную информацию
    debug_img = image.copy()
    for i, (x,y,w,h) in enumerate(boxes):
        cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(debug_img, f"Box {i}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite("debug_detection.jpg", debug_img)
    
    results = []
    for i, (x,y,w,h) in enumerate(boxes):
        crop = image[y:y+h, x:x+w]
        class_name = list(constants.TARGET_CLASSES.keys())[i]  # Получаем имя класса
        text = ocr_processor.recognize_text(crop, class_name)
        
        # Валидация только для основного класса
        if class_name == 'cn-11':
            # В цикле обработки результатов:
            if ContainerValidator.validate(text):
                results.append(text)
                print(f"Valid container: {text}")
            else:
                print(f"Invalid: {text} -> {ContainerValidator.validate_details(text)}")
        else:
            print(f"Additional info [{class_name}]: {text}")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    
    try:
        container_numbers = main(sys.argv[1])
        print("Valid container numbers found:")
        for num in container_numbers:
            print(f"- {num}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
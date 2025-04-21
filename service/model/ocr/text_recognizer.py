import easyocr
from service.model.config import constants
from ..processing.image_processing import ImageProcessor

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(
            ['en'], 
            gpu=True,
            model_storage_directory='models/',  # Путь для кэша моделей
            download_enabled=True
        )

    def recognize_text(self, image, class_name):
        preprocessed = ImageProcessor.preprocess_for_ocr(image, class_name)
        results = self.reader.readtext(
            preprocessed,
            allowlist=constants.ALLOWED_CHARS,
            decoder='beamsearch',
            batch_size=5,
            detail=0
        )
        return self.postprocess_text(' '.join(results), class_name)

    def postprocess_text(self, text, class_name):
    # Удаление пробелов и лишних символов
        clean_text = ''.join(filter(str.isalnum, text)).upper()
        
        # Обрезка до 11 символов для основного класса
        if class_name == 'cn-11':
            return clean_text[:11]
        
        return clean_text
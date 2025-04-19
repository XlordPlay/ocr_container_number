import easyocr
from config import constants

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

    def recognize_text(self, image):
        processed = ImageProcessor.preprocess_for_ocr(image)
        results = self.reader.readtext(
            processed,
            allowlist=constants.ALLOWED_CHARS,
            decoder='beamsearch',
            batch_size=5
        )
        return ' '.join([res[1] for res in results]).strip()
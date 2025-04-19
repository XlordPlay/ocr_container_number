import cv2

class ImageProcessor:
    @staticmethod
    def crop_boxes(image, boxes):
        cropped_images = []
        for box in boxes:
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            cropped = image[y:y+h, x:x+w]
            cropped_images.append(cropped)
        return cropped_images

    @staticmethod
    def preprocess_for_ocr(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.medianBlur(enhanced, 3)
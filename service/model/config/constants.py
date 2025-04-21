YOLO_WEIGHTS_PATH = '/home/xlordplay/bboom/ocr_container_number/data/weights/weights.pt'
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CONTAINER_PATTERN = r'^[A-Z]{4}\d{6}\d$'

# Добавляем фильтрацию по нужным классам
TARGET_CLASSES = {
    'cn-11': 0,  # Основной класс номера контейнера
    'cn-7': 1,   # Дополнительные классы (при необходимости)
    'cn-4': 2
}

# Остальные константы...
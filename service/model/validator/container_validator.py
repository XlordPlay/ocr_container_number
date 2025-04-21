import re
from service.model.config import constants

class ContainerValidator:
    @staticmethod
    def validate_format(text):
        """Проверка соответствия формату ISO 6346"""
        return re.fullmatch(constants.CONTAINER_PATTERN, text) is not None

    @staticmethod
    def calculate_checksum(text):
        """Расчет контрольной суммы"""
        letter_map = {chr(65+i): 10+i for i in range(26)}
        total = 0
        for i, char in enumerate(text[:10]):
            value = letter_map[char] if char.isalpha() else int(char)
            total += value * (2 ** i)
        return total % 11

    @staticmethod
    def validate_checksum(text):
        """Валидация контрольной суммы"""
        if len(text) != 11:
            return False
        return ContainerValidator.calculate_checksum(text) == int(text[-1])

    @staticmethod
    def validate(text):
        if len(text) != 11:
            return False
        return (
            ContainerValidator.validate_format(text) 
            and ContainerValidator.validate_checksum(text)
        )

    @staticmethod
    def validate_details(text):
        """Детальная информация о валидации"""
        errors = []
        if not ContainerValidator.validate_format(text):
            errors.append("Несоответствие формату (XXXX0000000)")
        if not ContainerValidator.validate_checksum(text):
            errors.append("Ошибка контрольной суммы")
        return "OK" if not errors else " | ".join(errors)
"""Лабораторная работа «Структурированный вывод LLM», уровень 3.

Реализуйте публичный интерфейс ниже. Внутреннюю структуру решения
спроектируйте самостоятельно.
"""

SYSTEM_PROMPT = ""


class ExtractionError(Exception):
    """Ответ модели не удалось превратить в корректный заказ."""


def build_messages(text):
    raise NotImplementedError


def parse_response(content):
    raise NotImplementedError


def validate_order(data):
    raise NotImplementedError


def extract_order(llm, text, max_retries=2):
    raise NotImplementedError

"""Лабораторная работа «Структурированный вывод LLM», уровень 2.

``llm.chat(messages)`` возвращает ответ модели; текст — в ``response.content``.
"""

import json
import re

SYSTEM_PROMPT = ""  # TODO: роль модели и формат ответа (JSON с полями product, quantity, city)
REQUIRED_FIELDS = ("product", "quantity", "city")


class ExtractionError(Exception):
    """Ответ модели не удалось превратить в корректный заказ."""


def build_messages(text):
    """Системное сообщение с SYSTEM_PROMPT и сообщение пользователя с текстом."""
    raise NotImplementedError


def parse_response(content):
    """Найти JSON-объект в ответе модели и вернуть словарь.

    JSON может быть обёрнут в блок ```json ... ``` или окружён текстом.
    Если JSON нет или он некорректен — выбросить ExtractionError.
    """
    raise NotImplementedError


def validate_order(data):
    """Проверить поля заказа и вернуть нормализованный словарь.

    Подсказка: quantity приводится к int, строки обрезаются через strip().
    """
    raise NotImplementedError


def extract_order(llm, text):
    """Один вызов модели → разбор → проверка."""
    raise NotImplementedError

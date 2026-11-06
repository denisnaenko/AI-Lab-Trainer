"""Лабораторная работа «Структурированный вывод LLM», уровень 1.

Модель передаётся в функцию как объект ``llm``:
``llm.chat(messages)`` возвращает ответ, текст которого лежит в ``response.content``.
"""

import json

# TODO: опишите роль модели и формат ответа — только JSON-объект
#       с полями product (строка), quantity (целое число) и city (строка).
SYSTEM_PROMPT = ""


def build_messages(text):
    """Вернуть список сообщений для модели: системное и пользовательское."""
    # TODO: первое сообщение — {"role": "system", "content": SYSTEM_PROMPT}
    # TODO: второе сообщение — {"role": "user", "content": text}
    raise NotImplementedError


def parse_response(content):
    """Превратить текст ответа модели (JSON) в словарь."""
    # TODO: используйте json.loads(content)
    raise NotImplementedError


def extract_order(llm, text):
    """Извлечь из сообщения клиента данные заказа с помощью модели."""
    # TODO: 1. Соберите сообщения функцией build_messages(text).
    # TODO: 2. Вызовите модель один раз: response = llm.chat(messages).
    # TODO: 3. Разберите response.content функцией parse_response и верните словарь.
    raise NotImplementedError

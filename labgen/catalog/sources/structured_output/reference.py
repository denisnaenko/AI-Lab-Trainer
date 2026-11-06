"""Эталонное решение: извлечение данных заказа из сообщения клиента с помощью LLM."""

import json
import re

SYSTEM_PROMPT = (
    "Ты — сервис извлечения данных из сообщений клиентов интернет-магазина. "
    "Найди в сообщении товар, количество и город доставки. "
    "Ответь только JSON-объектом с полями product (строка), quantity (целое число) и city (строка)."
)
REQUIRED_FIELDS = ("product", "quantity", "city")


class ExtractionError(Exception):
    """Ответ модели не удалось превратить в корректный заказ."""


def build_messages(text):
    """Сообщения для модели: системная инструкция и текст клиента."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_response(content):
    """Достаёт JSON-объект из ответа модели (в том числе из блока кода или текста)."""
    if not content:
        raise ExtractionError("пустой ответ модели")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end < start:
            raise ExtractionError("в ответе нет JSON-объекта")
        raw = content[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"некорректный JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ExtractionError("ожидался JSON-объект")
    return data


def validate_order(data):
    """Проверяет поля заказа и приводит их к нужным типам."""
    missing = [name for name in REQUIRED_FIELDS if data.get(name) in (None, "")]
    if missing:
        raise ExtractionError("не хватает полей: " + ", ".join(missing))
    try:
        quantity = int(data["quantity"])
    except (TypeError, ValueError):
        raise ExtractionError("поле quantity должно быть целым числом") from None
    if quantity <= 0:
        raise ExtractionError("поле quantity должно быть положительным")
    return {
        "product": str(data["product"]).strip(),
        "quantity": quantity,
        "city": str(data["city"]).strip(),
    }


def extract_order(llm, text, max_retries=2):
    """Извлекает заказ; при некорректном ответе просит модель исправиться."""
    messages = build_messages(text)
    last_error = None
    for _ in range(max_retries + 1):
        response = llm.chat(messages)
        try:
            return validate_order(parse_response(response.content))
        except ExtractionError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": response.content or ""},
                {"role": "user", "content": f"Ошибка: {exc}. Исправь ответ и верни только корректный JSON."},
            ]
    raise ExtractionError(f"модель не вернула корректный ответ: {last_error}")

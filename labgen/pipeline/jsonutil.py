from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (чистый JSON, блок ```json или JSON внутри текста)."""
    if not text or not text.strip():
        raise ValueError("пустой ответ модели")
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"в ответе модели нет корректного JSON-объекта: {last_error}")


def extract_python_code(text: str) -> str:
    """Код из блока ```python ... ``` (или весь текст, если блока нет)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"

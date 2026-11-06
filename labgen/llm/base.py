from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Ошибка обращения к модели."""


class LLMProvider(Protocol):
    """Минимальный интерфейс модели для конвейера: сообщения на вход, текст на выход."""

    name: str

    def complete(self, messages: list[dict], *, temperature: float | None = None, json_mode: bool = False) -> str:
        ...

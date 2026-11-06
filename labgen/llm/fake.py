"""Модель-«пустышка» для тестов и демонстраций конвейера без доступа к API."""

from __future__ import annotations

from typing import Callable, Iterable, Union

Reply = Union[str, Callable[[list[dict]], str], Exception]


class ReplayProvider:
    """Возвращает заранее заданные ответы по очереди и запоминает запросы."""

    def __init__(self, replies: Iterable[Reply], name: str = "replay") -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.name = name

    def complete(self, messages: list[dict], *, temperature: float | None = None, json_mode: bool = False) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise RuntimeError("ReplayProvider: ответы закончились")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages)
        return reply

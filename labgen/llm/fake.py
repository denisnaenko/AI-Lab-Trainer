"""Тестовая чат-модель LangChain: отдаёт заданные ответы и записывает запросы."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .factory import LabLLM

_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class RecordingChatModel(BaseChatModel):
    """Ответ — строка, исключение или функция ``f(messages: list[dict]) -> str``."""

    replies: list[Any] = []
    calls: list[list[dict]] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        recorded = [{"role": _ROLES.get(m.type, m.type), "content": m.content} for m in messages]
        self.calls.append(recorded)
        if not self.replies:
            raise RuntimeError("RecordingChatModel: ответы закончились")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(recorded)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def fake_llm(replies: Iterable[str | Exception | Callable[[list[dict]], str]], name: str = "fake") -> LabLLM:
    """LabLLM поверх тестовой модели; запросы доступны в ``llm.chat.calls``."""
    return LabLLM(chat=RecordingChatModel(replies=list(replies), calls=[]), name=name)

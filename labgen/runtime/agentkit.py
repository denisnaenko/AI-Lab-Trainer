"""agentkit — минимальная среда для лабораторных работ по агентной разработке.

Студенческий агент получает объект ``llm`` с методом ``chat(messages, tools=None)``,
который возвращает :class:`LLMResponse`. В тестах вместо настоящей модели
используется :class:`ScriptedLLM` — «LLM по сценарию»: она отдаёт заранее
заданные ответы и записывает все запросы агента. Поэтому поведение агента
можно проверять детерминированно: какие инструменты он вызвал, с какими
аргументами, что отправил модели и когда остановился.

Формат сообщений совместим с OpenAI Chat Completions::

    {"role": "system" | "user" | "assistant" | "tool", "content": "..."}

Схема инструмента (tool schema) — тоже в формате OpenAI::

    {"type": "function",
     "function": {"name": "get_weather", "description": "...",
                  "parameters": {"type": "object", "properties": ..., "required": [...]}}}
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "ToolCall",
    "LLMResponse",
    "ScriptedLLM",
    "ScriptExhausted",
    "OpenAICompatibleLLM",
    "text",
    "call",
    "tool_call",
    "tool_calls",
    "assistant_message",
    "tool_result_message",
]

_ids = itertools.count(1)


@dataclass
class ToolCall:
    """Запрос модели на вызов инструмента.

    ``arguments`` — словарь; настоящие API иногда присылают JSON-строку,
    поэтому надёжный агент должен уметь обработать и её.
    """

    name: str
    arguments: Any = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"call_{next(_ids)}"


@dataclass
class LLMResponse:
    """Ответ модели: текст и/или список запросов на вызов инструментов."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


# --- конструкторы ответов для сценариев -------------------------------------


def text(content: str) -> LLMResponse:
    """Ответ модели обычным текстом."""
    return LLMResponse(content=content)


def call(name: str, **arguments: Any) -> ToolCall:
    """Один запрос на вызов инструмента (для :func:`tool_calls`)."""
    return ToolCall(name=name, arguments=arguments)


def tool_call(name: str, **arguments: Any) -> LLMResponse:
    """Ответ модели с одним вызовом инструмента."""
    return LLMResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])


def tool_calls(*calls: ToolCall) -> LLMResponse:
    """Ответ модели с несколькими вызовами инструментов сразу."""
    return LLMResponse(tool_calls=list(calls))


# --- сообщения -----------------------------------------------------------------


def assistant_message(response: LLMResponse) -> dict:
    """Сообщение ассистента для истории диалога (с запросами инструментов, если они есть)."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls
        ]
    return message


def tool_result_message(tool_call_: ToolCall, content: Any) -> dict:
    """Сообщение с результатом инструмента, привязанное к запросу по ``tool_call_id``."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_.id,
        "name": tool_call_.name,
        "content": str(content),
    }


# --- LLM по сценарию -------------------------------------------------------------


class ScriptExhausted(RuntimeError):
    """Агент обратился к модели больше раз, чем предусмотрено сценарием."""


class ScriptedLLM:
    """Детерминированная «модель» для тестов.

    Элемент сценария может быть:

    * строкой — ответ текстом;
    * :class:`LLMResponse` — любой ответ, в том числе с вызовами инструментов;
    * исключением — оно будет выброшено (имитация сбоя API);
    * функцией ``f(messages, tools) -> LLMResponse | str`` — ответ, зависящий от запроса.

    Все запросы сохраняются в ``calls``: ``[{"messages": [...], "tools": [...]}, ...]``.
    """

    def __init__(self, responses: Iterable[Any] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self.responses:
            raise ScriptExhausted(
                f"ScriptedLLM: сценарий закончился на вызове №{len(self.calls)} — "
                "агент обращается к модели чаще, чем ожидалось"
            )
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(copy.deepcopy(messages), copy.deepcopy(tools))
        if isinstance(item, str):
            item = LLMResponse(content=item)
        if not isinstance(item, LLMResponse):
            raise TypeError(f"ScriptedLLM: неподдерживаемый элемент сценария: {item!r}")
        return copy.deepcopy(item)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def messages_of(self, index: int) -> list[dict]:
        """Сообщения, отправленные модели в вызове ``index`` (0 — первый вызов)."""
        return self.calls[index]["messages"]


# --- настоящая модель (для экспериментов студента) -------------------------------


class OpenAICompatibleLLM:
    """Клиент для OpenAI-совместимого API (OpenAI, Ollama, vLLM и др.).

    Параметры берутся из аргументов или переменных окружения
    ``AGENTKIT_BASE_URL``, ``AGENTKIT_API_KEY``, ``AGENTKIT_MODEL``.
    В автотестах не используется.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or os.environ.get("AGENTKIT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("AGENTKIT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTKIT_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature

    @staticmethod
    def _to_openai(message: dict) -> dict:
        result = {k: v for k, v in message.items() if k in ("role", "content", "tool_call_id", "name")}
        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"], ensure_ascii=False),
                    },
                }
                for c in message["tool_calls"]
            ]
        return result

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            arguments: Any = raw["function"].get("arguments", "")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) and arguments else arguments
            except json.JSONDecodeError:
                pass
            calls.append(ToolCall(name=raw["function"]["name"], arguments=arguments, id=raw.get("id", "")))
        return LLMResponse(content=message.get("content"), tool_calls=calls)

"""Подключение моделей через LangChain.

Поддерживаются три вида интеграций:

* ``anthropic`` — Claude через ``langchain_anthropic.ChatAnthropic``;
* ``openai`` — OpenAI через ``langchain_openai.ChatOpenAI``;
* ``local`` — локальная модель с OpenAI-совместимым API (Ollama, LM Studio, vLLM)
  через ``ChatOpenAI`` с собственным ``base_url``.

Конвейер работает с :class:`LabLLM` — тонкой обёрткой над ``BaseChatModel``,
которая знает, как попросить у модели JSON, и хранит подпись модели для журнала.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class LLMError(RuntimeError):
    """Ошибка обращения к модели."""


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    title: str
    default_model: str
    needs_key: bool
    needs_url: bool
    default_url: str = ""
    model_hint: str = ""
    key_hint: str = ""


PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo(
        id="anthropic",
        title="Claude (Anthropic)",
        default_model="claude-sonnet-5-5",
        needs_key=True,
        needs_url=False,
        model_hint="например, claude-sonnet-5-5 или claude-opus-5-5",
        key_hint="ключ из console.anthropic.com, начинается с sk-ant-",
    ),
    "openai": ProviderInfo(
        id="openai",
        title="OpenAI",
        default_model="gpt-5-mini",
        needs_key=True,
        needs_url=False,
        model_hint="например, gpt-5-mini или gpt-5",
        key_hint="ключ из platform.openai.com, начинается с sk-",
    ),
    "local": ProviderInfo(
        id="local",
        title="Локальная LLM",
        default_model="qwen2.5-coder:14b",
        needs_key=False,
        needs_url=True,
        default_url="http://localhost:11434/v1",
        model_hint="имя модели на вашем сервере, например qwen2.5-coder:14b",
        key_hint="обычно не нужен; укажите, если сервер требует токен",
    ),
}


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    json_mode: bool = False
    temperature: float | None = None
    timeout: float = 240.0
    max_tokens: int = 16000

    @property
    def label(self) -> str:
        title = PROVIDERS[self.provider].title if self.provider in PROVIDERS else self.provider
        return f"{title} · {self.model}"


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """Создаёт чат-модель LangChain по настройкам интеграции."""
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not config.api_key:
            raise LLMError("Для Claude нужен API-ключ")
        kwargs: dict[str, Any] = dict(
            model=config.model, api_key=config.api_key, max_tokens=config.max_tokens,
            timeout=config.timeout, max_retries=2,
        )
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        return ChatAnthropic(**kwargs)

    if config.provider in ("openai", "local"):
        from langchain_openai import ChatOpenAI

        if config.provider == "openai" and not config.api_key:
            raise LLMError("Для OpenAI нужен API-ключ")
        if config.provider == "local" and not config.base_url:
            raise LLMError("Для локальной LLM нужен адрес сервера (base URL)")
        kwargs = dict(
            model=config.model,
            api_key=config.api_key or "not-needed",
            timeout=config.timeout,
            max_retries=2,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url.rstrip("/")
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        return ChatOpenAI(**kwargs)

    raise LLMError(f"Неизвестный провайдер LLM: {config.provider}")


def to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # ответы некоторых моделей — список блоков (текст, рассуждения и т. п.)
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


@dataclass
class LabLLM:
    """Модель для конвейера: обычный и JSON-режим поверх одной чат-модели LangChain."""

    chat: BaseChatModel
    name: str = "llm"
    json_mode: bool = False
    usage: dict = field(default_factory=lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0})

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LabLLM":
        json_mode = config.json_mode and config.provider in ("openai", "local")
        return cls(chat=build_chat_model(config), name=config.label, json_mode=json_mode)

    def complete(self, messages: list[dict], *, json: bool = False) -> str:
        runnable = self.chat
        if json and self.json_mode:
            runnable = self.chat.bind(response_format={"type": "json_object"})
        try:
            message = runnable.invoke(to_lc_messages(messages))
        except Exception as exc:  # ошибки SDK провайдеров разнородны
            raise LLMError(_safe_error(exc)) from exc
        self.usage["requests"] += 1
        meta = getattr(message, "usage_metadata", None) or {}
        self.usage["input_tokens"] += int(meta.get("input_tokens", 0) or 0)
        self.usage["output_tokens"] += int(meta.get("output_tokens", 0) or 0)
        return _text(message)


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    # не показываем ключи, даже если SDK вставил их в сообщение
    import re

    text = re.sub(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]+", r"\1…", text)
    return text[:500]


def check_connection(config: LLMConfig) -> tuple[bool, str]:
    """Короткий запрос к модели: проверка ключа, адреса и имени модели."""
    try:
        probe = LLMConfig(**{**config.__dict__, "max_tokens": 16, "timeout": 30.0})
        llm = LabLLM.from_config(probe)
        answer = llm.complete([{"role": "user", "content": "Ответь одним словом: OK"}])
    except LLMError as exc:
        return False, str(exc)
    return True, f"Модель ответила: «{answer.strip()[:60]}»"

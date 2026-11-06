"""Модели LLM через LangChain: Claude, OpenAI и локальные OpenAI-совместимые серверы."""

from __future__ import annotations

from ..config import Settings
from .factory import (
    PROVIDERS,
    LabLLM,
    LLMConfig,
    LLMError,
    ProviderInfo,
    build_chat_model,
    check_connection,
)
from .fake import RecordingChatModel, fake_llm

__all__ = [
    "PROVIDERS",
    "LabLLM",
    "LLMConfig",
    "LLMError",
    "ProviderInfo",
    "RecordingChatModel",
    "build_chat_model",
    "check_connection",
    "config_from_settings",
    "fake_llm",
]


def config_from_settings(settings: Settings) -> LLMConfig | None:
    """Настройки LLM из окружения (для CLI). ``None`` — офлайн-режим."""
    provider = settings.llm_provider or ("local" if settings.llm_base_url else "")
    if not provider:
        return None
    info = PROVIDERS.get(provider)
    if info is None:
        raise LLMError(f"LABGEN_LLM_PROVIDER: неизвестный провайдер {provider!r}")
    return LLMConfig(
        provider=provider,
        model=settings.llm_model or info.default_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or info.default_url,
        json_mode=settings.llm_json_mode,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )

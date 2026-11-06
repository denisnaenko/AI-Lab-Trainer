"""Клиенты LLM для конвейера генерации."""

from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider
from .fake import ReplayProvider
from .openai_compat import OpenAICompatibleProvider

__all__ = ["LLMError", "LLMProvider", "ReplayProvider", "OpenAICompatibleProvider", "provider_from_settings"]


def provider_from_settings(settings: Settings) -> LLMProvider | None:
    """Клиент LLM по настройкам или ``None``, если модель не настроена (офлайн-режим)."""
    if not settings.llm_configured:
        return None
    return OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
        json_mode=settings.llm_json_mode,
        temperature=settings.llm_temperature,
    )

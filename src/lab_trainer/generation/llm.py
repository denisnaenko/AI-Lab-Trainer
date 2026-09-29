"""Chat model factory. Any LangChain chat model that supports `with_structured_output` works.

The model is chosen with a `provider:model` string (LangChain's `init_chat_model` format):

    anthropic:claude-opus-5-5        default; needs ANTHROPIC_API_KEY
    openai:gpt-5                     needs OPENAI_API_KEY
    openai:qwen3-coder               any OpenAI-compatible server (Ollama, vLLM, LM Studio):
                                     set LAB_TRAINER_BASE_URL, e.g. http://localhost:11434/v1

Tests pass a scripted fake instead, so no generation test needs an API key.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


def make_chat_model(model: str | None = None, base_url: str | None = None) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    model = model or os.environ.get("LAB_TRAINER_MODEL", DEFAULT_MODEL)
    base_url = base_url or os.environ.get("LAB_TRAINER_BASE_URL")
    kwargs: dict = {"max_tokens": 16000}
    if base_url:
        kwargs["base_url"] = base_url
        # local OpenAI-compatible servers usually ignore the key, but the client requires one
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "not-needed")
    return init_chat_model(model, **kwargs)

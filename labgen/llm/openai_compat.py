"""Клиент для OpenAI-совместимого Chat Completions API.

Подходит для OpenAI и для локальных серверов с совместимым API (Ollama, vLLM, LM Studio),
а также для облачных моделей, которые предоставляют совместимую точку доступа.
"""

from __future__ import annotations

import time

import httpx

from .base import LLMError


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 180.0,
        json_mode: bool = True,
        temperature: float = 0.2,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.json_mode = json_mode
        self.temperature = temperature
        self.retries = retries
        self.name = f"openai-compatible:{model}"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def complete(self, messages: list[dict], *, temperature: float | None = None, json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    # ошибки запроса не исправятся повтором
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
                data = response.json()
                usage = data.get("usage") or {}
                self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                self.usage["requests"] += 1
                content = data["choices"][0]["message"].get("content") or ""
                return content
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as exc:
                last_error = exc
                retriable = isinstance(exc, httpx.HTTPError) or (
                    isinstance(exc, LLMError) and str(exc).startswith(("HTTP 429", "HTTP 5"))
                )
                if not retriable or attempt == self.retries:
                    break
                time.sleep(2 ** attempt)
        raise LLMError(f"Не удалось получить ответ модели {self.model}: {last_error}")

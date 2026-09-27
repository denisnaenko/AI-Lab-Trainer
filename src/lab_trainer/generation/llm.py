"""Thin wrapper over the Anthropic SDK so the pipeline can be tested with a fake."""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("LAB_TRAINER_MODEL", "claude-opus-5")


class LLMClient(Protocol):
    def structured(self, system: str, prompt: str, schema: type[T]) -> T: ...


class ClaudeClient:
    """Structured-output calls: the response is validated against a Pydantic schema."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {response.stop_details}")
        return response.parsed_output

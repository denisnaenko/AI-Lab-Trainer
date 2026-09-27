import pytest


class FakeLLM:
    """Scripted stand-in for an LLM: returns the given responses in order, records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return {"type": "tool_call", "name": "calculator", "arguments": {"expression": "1+1"}}
        return self.responses.pop(0)


@pytest.fixture
def fake_llm():
    return FakeLLM

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ids = itertools.count(1)


class FakeLLM(BaseChatModel):
    """Scripted LangChain chat model: returns the given AIMessages in order, records calls.

    When the script runs out it keeps asking for the calculator tool, so an agent without
    a step limit would never stop.
    """

    responses: list = []
    calls: list = []
    bound_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    @staticmethod
    def say(text):
        return AIMessage(content=text)

    @staticmethod
    def call(name, **args):
        return AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}]
        )

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", getattr(t, "__name__", t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        message = self.responses.pop(0) if self.responses else self.call(
            "calculator", expression="1+1"
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def fake_llm():
    def make(responses):
        return FakeLLM(responses=list(responses), calls=[])

    make.say = FakeLLM.say
    make.call = FakeLLM.call
    return make

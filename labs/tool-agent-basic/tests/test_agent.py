import pytest
from langchain_core.messages import ToolMessage

from agent import calculator, run_agent


@pytest.mark.parametrize("expr, expected", [
    ("2+2", 4), ("2*(3+4)", 14), ("10/4", 2.5), ("7-10", -3), ("(1+2)*(3+4)/7", 3),
])
def test_calculator(expr, expected):
    assert float(calculator.invoke({"expression": expr})) == pytest.approx(expected)


def test_direct_answer(fake_llm):
    llm = fake_llm([fake_llm.say("Привет!")])
    assert run_agent(llm, "Привет") == "Привет!"


def test_tool_call_roundtrip(fake_llm):
    request = fake_llm.call("calculator", expression="6*7")
    llm = fake_llm([request, fake_llm.say("42")])
    assert run_agent(llm, "Сколько будет 6*7?") == "42"
    assert "calculator" in llm.bound_tools
    tool_msgs = [m for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[-1].tool_call_id == request.tool_calls[0]["id"]
    assert float(tool_msgs[-1].content) == 42


def test_max_steps(fake_llm):
    llm = fake_llm([])  # always asks for a tool, never answers
    result = run_agent(llm, "loop", max_steps=3)
    assert len(llm.calls) == 3
    assert result.startswith("Stopped")


def test_unknown_tool(fake_llm):
    llm = fake_llm([fake_llm.call("weather"), fake_llm.say("Не могу")])
    assert run_agent(llm, "Погода?") == "Не могу"
    tool_msgs = [m for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "weather" in tool_msgs[-1].content

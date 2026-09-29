"""Reference solution for tool-agent-basic."""

import ast
import operator

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> float:
    """Вычисляет арифметическое выражение с + - * / и скобками."""
    return _eval(ast.parse(expression, mode="eval").body)


TOOLS = [calculator]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_agent(llm, user_message: str, max_steps: int = 5) -> str:
    model = llm.bind_tools(TOOLS)
    messages = [HumanMessage(user_message)]
    for _ in range(max_steps):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            try:
                result = str(TOOLS_BY_NAME[call["name"]].invoke(call["args"]))
            except KeyError:
                result = f"Error: unknown tool {call['name']}"
            except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
                result = f"Error: {exc}"
            messages.append(ToolMessage(result, name=call["name"], tool_call_id=call["id"]))
    return f"Stopped after {max_steps} steps"

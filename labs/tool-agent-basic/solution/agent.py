"""Reference solution for tool-agent-basic."""

import ast
import operator

TOOLS = [
    {
        "name": "calculator",
        "description": "Вычисляет арифметическое выражение с + - * / и скобками",
        "parameters": {"expression": "string"},
    }
]

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


def calculator(expression: str) -> float:
    return _eval(ast.parse(expression, mode="eval").body)


TOOL_FUNCTIONS = {"calculator": calculator}


def run_agent(llm, user_message: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm.complete(messages, tools=TOOLS)
        if response["type"] == "text":
            return response["text"]
        name = response["name"]
        messages.append({"role": "assistant", "tool_call": response})
        try:
            result = str(TOOL_FUNCTIONS[name](**response.get("arguments", {})))
        except KeyError:
            result = f"Error: unknown tool {name}"
        except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
            result = f"Error: {exc}"
        messages.append({"role": "tool", "name": name, "content": result})
    return f"Stopped after {max_steps} steps"

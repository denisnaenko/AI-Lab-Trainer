"""Лабораторная: агент с инструментами. Заполните места, помеченные TODO."""

TOOLS = [
    {
        "name": "calculator",
        "description": "Вычисляет арифметическое выражение с + - * / и скобками",
        "parameters": {"expression": "string"},
    }
]


def calculator(expression: str) -> float:
    # TODO: вычислите выражение без прямого eval (например, через модуль ast)
    raise NotImplementedError


TOOL_FUNCTIONS = {"calculator": calculator}


def run_agent(llm, user_message: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm.complete(messages, tools=TOOLS)
        # TODO: если ответ текстовый — вернуть его
        # TODO: если это вызов инструмента — выполнить его и добавить сообщение role=tool
        raise NotImplementedError
    return "Stopped"

"""Лабораторная: агент с инструментами на LangChain. Заполните места, помеченные TODO."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> float:
    """Вычисляет арифметическое выражение с + - * / и скобками."""
    # TODO: вычислите выражение без прямого eval (например, через модуль ast)
    raise NotImplementedError


TOOLS = [calculator]


def run_agent(llm, user_message: str, max_steps: int = 5) -> str:
    """llm — чат-модель LangChain (BaseChatModel)."""
    model = llm.bind_tools(TOOLS)
    messages = [HumanMessage(user_message)]
    for _ in range(max_steps):
        response = model.invoke(messages)
        # TODO: если в ответе нет tool_calls — вернуть response.content
        # TODO: иначе выполнить каждый вызов и добавить ToolMessage с тем же tool_call_id
        raise NotImplementedError
    return "Stopped"

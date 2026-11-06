"""Лабораторная работа «Агент с инструментами», уровень 2.

``llm.chat(messages, tools=...)`` возвращает ответ с полями ``content`` и ``tool_calls``.
Помощники: ``assistant_message(response)``, ``tool_result_message(call, result)``.
"""

from agentkit import assistant_message, tool_result_message

DEFAULT_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Если для ответа нужны данные, вызови подходящий инструмент. "
    "Когда данных достаточно, ответь пользователю текстом."
)
STEP_LIMIT_MESSAGE = "Не удалось получить ответ: превышено число шагов агента."


class ToolAgent:
    def __init__(self, llm, tools, tool_schemas, system_prompt=DEFAULT_SYSTEM_PROMPT, max_steps=5):
        self.llm = llm
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_message):
        """Цикл «модель → инструменты → модель» не длиннее max_steps вызовов модели.

        Подсказка: модель может запросить несколько инструментов сразу,
        а может — инструмент, которого нет в self.tools.
        """
        raise NotImplementedError

"""Лабораторная работа «Агент с инструментами», уровень 3.

Реализуйте публичный интерфейс ниже; внутреннюю структуру спроектируйте сами.
"""

DEFAULT_SYSTEM_PROMPT = "Ты — полезный ассистент, который умеет вызывать инструменты."
STEP_LIMIT_MESSAGE = "Не удалось получить ответ: превышено число шагов агента."


class ToolAgent:
    def __init__(self, llm, tools, tool_schemas, system_prompt=DEFAULT_SYSTEM_PROMPT, max_steps=5):
        self.trace = []
        raise NotImplementedError

    def run(self, user_message):
        raise NotImplementedError

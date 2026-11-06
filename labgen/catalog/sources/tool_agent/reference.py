"""Эталонное решение: агент, который вызывает инструменты по запросу модели."""

import json

from agentkit import assistant_message, tool_result_message

DEFAULT_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Если для ответа нужны данные, вызови подходящий инструмент. "
    "Когда данных достаточно, ответь пользователю текстом."
)
STEP_LIMIT_MESSAGE = "Не удалось получить ответ: превышено число шагов агента."


class ToolAgent:
    """Агент с циклом «модель → инструменты → модель»."""

    def __init__(self, llm, tools, tool_schemas, system_prompt=DEFAULT_SYSTEM_PROMPT, max_steps=5):
        self.llm = llm
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.trace = []

    def _parse_arguments(self, arguments):
        if isinstance(arguments, str):
            return json.loads(arguments) if arguments.strip() else {}
        return arguments or {}

    def _execute(self, call):
        """Выполняет вызов инструмента и возвращает (аргументы, текст результата)."""
        try:
            arguments = self._parse_arguments(call.arguments)
        except json.JSONDecodeError as exc:
            return call.arguments, f"Ошибка: некорректные аргументы JSON ({exc.msg})"
        if call.name not in self.tools:
            return arguments, f"Неизвестный инструмент: {call.name}"
        try:
            result = self.tools[call.name](**arguments)
        except Exception as exc:
            return arguments, f"Ошибка: {exc}"
        return arguments, str(result)

    def run(self, user_message):
        """Отвечает на запрос пользователя, вызывая инструменты по просьбе модели."""
        self.trace = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        for _ in range(self.max_steps):
            response = self.llm.chat(messages, tools=self.tool_schemas)
            if not response.tool_calls:
                return response.content or ""
            messages.append(assistant_message(response))
            for call in response.tool_calls:
                arguments, result = self._execute(call)
                self.trace.append({"tool": call.name, "arguments": arguments, "result": result})
                messages.append(tool_result_message(call, result))
        return STEP_LIMIT_MESSAGE

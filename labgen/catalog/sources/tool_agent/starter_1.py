"""Лабораторная работа «Агент с инструментами», уровень 1.

Модель: ``response = llm.chat(messages, tools=tool_schemas)``.
* Если ``response.tool_calls`` пуст — в ``response.content`` готовый ответ.
* Иначе каждый элемент ``response.tool_calls`` — запрос на вызов инструмента
  с полями ``name``, ``arguments`` (словарь) и ``id``.

Готовые помощники из agentkit:
* ``assistant_message(response)`` — сообщение ассистента для истории;
* ``tool_result_message(call, result)`` — сообщение с результатом инструмента.
"""

from agentkit import assistant_message, tool_result_message

DEFAULT_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Если для ответа нужны данные, вызови подходящий инструмент. "
    "Когда данных достаточно, ответь пользователю текстом."
)


# Пример инструмента и его схемы для ручных экспериментов.
def get_weather(city):
    return {"Москва": "+3°C, облачно", "Казань": "-1°C, снег"}.get(city, "нет данных")


WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Текущая погода в городе",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


class ToolAgent:
    def __init__(self, llm, tools, tool_schemas, system_prompt=DEFAULT_SYSTEM_PROMPT):
        """tools — словарь {имя: функция}, tool_schemas — список схем для модели."""
        self.llm = llm
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt

    def run(self, user_message):
        """Ответить на запрос пользователя, вызывая инструменты по просьбе модели."""
        # TODO: 1. Создайте список сообщений: системное (self.system_prompt) и пользовательское.
        # TODO: 2. Вызовите модель: response = self.llm.chat(messages, tools=self.tool_schemas).
        # TODO: 3. Если в ответе нет tool_calls — верните response.content.
        # TODO: 4. Иначе добавьте в messages assistant_message(response),
        #          для каждого запроса вызовите функцию: self.tools[call.name](**call.arguments)
        #          и добавьте tool_result_message(call, result).
        # TODO: 5. Снова вызовите модель с обновлённым списком и верните её текстовый ответ.
        raise NotImplementedError

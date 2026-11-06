"""Лабораторная работа «Агент с памятью диалога», уровень 2.

Модель: ``response = llm.chat(messages)``, текст ответа — ``response.content``.
"""

DEFAULT_SYSTEM_PROMPT = "Ты — дружелюбный ассистент. Учитывай весь предыдущий диалог."


class ChatAgent:
    def __init__(self, llm, system_prompt=DEFAULT_SYSTEM_PROMPT, max_history=None):
        """max_history — сколько последних сообщений хранить (None — без ограничения)."""
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history = []

    def send(self, user_message):
        """Отправить сообщение и вернуть ответ модели; при необходимости обрезать историю."""
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

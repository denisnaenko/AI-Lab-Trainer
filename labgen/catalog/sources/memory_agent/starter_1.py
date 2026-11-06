"""Лабораторная работа «Агент с памятью диалога», уровень 1.

Модель: ``response = llm.chat(messages)``, текст ответа — ``response.content``.
"""

DEFAULT_SYSTEM_PROMPT = "Ты — дружелюбный ассистент. Учитывай весь предыдущий диалог."


class ChatAgent:
    def __init__(self, llm, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = []  # сообщения пользователя и ответы модели, без системного

    def send(self, user_message):
        """Отправить сообщение пользователя и вернуть ответ модели."""
        # TODO: 1. Соберите сообщения: системное, затем self.history, затем новое сообщение пользователя.
        # TODO: 2. Вызовите self.llm.chat(messages) и возьмите текст ответа.
        # TODO: 3. Добавьте в self.history сообщение пользователя и ответ ассистента
        #          ({"role": "assistant", "content": ...}).
        # TODO: 4. Верните текст ответа.
        raise NotImplementedError

    def reset(self):
        """Начать диалог заново."""
        # TODO: очистите историю
        raise NotImplementedError

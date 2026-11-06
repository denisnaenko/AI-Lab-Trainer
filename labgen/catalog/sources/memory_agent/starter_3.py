"""Лабораторная работа «Агент с памятью диалога», уровень 3.

Реализуйте публичный интерфейс ниже; внутреннюю структуру спроектируйте сами.
"""

DEFAULT_SYSTEM_PROMPT = "Ты — дружелюбный ассистент. Учитывай весь предыдущий диалог."
SUMMARY_PROMPT = (
    "Кратко перескажи диалог ниже: сохрани факты о пользователе, договорённости и открытые вопросы. "
    "Пиши от третьего лица, не больше пяти предложений."
)
SUMMARY_PREFIX = "Краткое содержание предыдущего диалога: "
KEEP_LAST = 2


class ChatAgent:
    def __init__(self, llm, system_prompt=DEFAULT_SYSTEM_PROMPT, max_history=None, summarize_threshold=None):
        self.history = []
        self.summary = None
        raise NotImplementedError

    def send(self, user_message):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

"""Лабораторная работа «Мультиагентная система», уровень 3.

Реализуйте публичный интерфейс ниже; внутреннюю структуру спроектируйте сами.
"""

ROUTER_PROMPT = "Ты — диспетчер обращений. Выбери ровно один маршрут из списка и ответь только его названием."
FALLBACK_ROUTE = "fallback"
FALLBACK_MESSAGE = "Не удалось определить, кто обработает запрос. Обращение передано оператору."
AGENT_ERROR_MESSAGE = "Исполнитель временно недоступен, попробуйте позже."
APPROVED = "ОДОБРЕНО"
WRITER_PROMPT = "Ты — автор учебных текстов. Пиши кратко и точно."
CRITIC_PROMPT = f"Ты — строгий рецензент. Если текст можно публиковать, начни ответ словом {APPROVED}."


class Router:
    def __init__(self, llm, routes):
        raise NotImplementedError

    def route(self, request):
        raise NotImplementedError


class MultiAgentSystem:
    def __init__(self, router, agents, fallback=None):
        self.log = []
        raise NotImplementedError

    def handle(self, request):
        raise NotImplementedError


class WriterCriticLoop:
    def __init__(self, writer_llm, critic_llm, max_rounds=3):
        raise NotImplementedError

    def run(self, task):
        raise NotImplementedError

"""Лабораторная работа «Мультиагентная система», уровень 1.

Модель-диспетчер: ``response = llm.chat(messages)``, текст ответа — ``response.content``.
Исполнитель — обычная функция ``agent(request) -> str``.
"""

ROUTER_PROMPT = "Ты — диспетчер обращений. Выбери ровно один маршрут из списка и ответь только его названием."


class Router:
    def __init__(self, llm, routes):
        """routes — словарь {название маршрута: описание}."""
        self.llm = llm
        self.routes = routes

    def route(self, request):
        """Спросить модель, какой маршрут подходит запросу, и вернуть его название."""
        # TODO: 1. Соберите строки вида "- billing: оплата и счета" для всех маршрутов.
        # TODO: 2. Отправьте модели системное сообщение ROUTER_PROMPT и сообщение пользователя
        #          со списком маршрутов и текстом запроса.
        # TODO: 3. Очистите ответ модели (strip, lower) и верните название маршрута.
        raise NotImplementedError


class MultiAgentSystem:
    def __init__(self, router, agents):
        """agents — словарь {название маршрута: функция-исполнитель}."""
        self.router = router
        self.agents = agents

    def handle(self, request):
        """Вернуть {"route": маршрут, "answer": ответ исполнителя}."""
        # TODO: определите маршрут через self.router и вызовите нужного исполнителя
        raise NotImplementedError

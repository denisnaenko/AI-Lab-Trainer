"""Лабораторная работа «Мультиагентная система», уровень 2."""

import re

ROUTER_PROMPT = "Ты — диспетчер обращений. Выбери ровно один маршрут из списка и ответь только его названием."
FALLBACK_ROUTE = "fallback"
FALLBACK_MESSAGE = "Не удалось определить, кто обработает запрос. Обращение передано оператору."


class Router:
    def __init__(self, llm, routes):
        self.llm = llm
        self.routes = routes

    def route(self, request):
        """Название маршрута или None, если ответ модели не удалось однозначно распознать."""
        raise NotImplementedError


class MultiAgentSystem:
    def __init__(self, router, agents, fallback=None):
        """fallback — необязательная функция для запросов без подходящего маршрута."""
        self.router = router
        self.agents = agents
        self.fallback = fallback
        self.log = []

    def handle(self, request):
        raise NotImplementedError

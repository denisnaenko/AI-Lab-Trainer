"""Эталонное решение: мультиагентная система с диспетчером и циклом «автор — критик»."""

import re

ROUTER_PROMPT = "Ты — диспетчер обращений. Выбери ровно один маршрут из списка и ответь только его названием."
FALLBACK_ROUTE = "fallback"
FALLBACK_MESSAGE = "Не удалось определить, кто обработает запрос. Обращение передано оператору."
AGENT_ERROR_MESSAGE = "Исполнитель временно недоступен, попробуйте позже."
APPROVED = "ОДОБРЕНО"
WRITER_PROMPT = "Ты — автор учебных текстов. Пиши кратко и точно."
CRITIC_PROMPT = (
    f"Ты — строгий рецензент учебных текстов. Если текст можно публиковать, начни ответ словом {APPROVED}. "
    "Иначе перечисли, что нужно исправить."
)


class Router:
    """Диспетчер: с помощью модели выбирает маршрут для запроса."""

    def __init__(self, llm, routes):
        self.llm = llm
        self.routes = routes

    def build_messages(self, request):
        lines = "\n".join(f"- {name}: {description}" for name, description in self.routes.items())
        return [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": f"Маршруты:\n{lines}\n\nЗапрос: {request}"},
        ]

    def route(self, request):
        response = self.llm.chat(self.build_messages(request))
        answer = (response.content or "").strip().lower()
        names = {name.lower(): name for name in self.routes}
        if answer in names:
            return names[answer]
        words = set(re.findall(r"\w+", answer))
        found = [name for key, name in names.items() if key in words]
        if len(found) == 1:
            return found[0]
        return None


class MultiAgentSystem:
    """Передаёт запрос исполнителю выбранного маршрута и ведёт журнал."""

    def __init__(self, router, agents, fallback=None):
        self.router = router
        self.agents = agents
        self.fallback = fallback
        self.log = []

    def handle(self, request):
        route = self.router.route(request)
        if route not in self.agents:
            route = FALLBACK_ROUTE
            answer = self.fallback(request) if self.fallback else FALLBACK_MESSAGE
            status = "fallback"
        else:
            try:
                answer = self.agents[route](request)
                status = "ok"
            except Exception as exc:
                answer = AGENT_ERROR_MESSAGE
                status = f"error: {exc}"
        self.log.append({"request": request, "route": route, "status": status})
        return {"route": route, "answer": answer}


class WriterCriticLoop:
    """Автор пишет текст, критик рецензирует, автор исправляет — до одобрения или лимита."""

    def __init__(self, writer_llm, critic_llm, max_rounds=3):
        self.writer_llm = writer_llm
        self.critic_llm = critic_llm
        self.max_rounds = max_rounds

    def _write(self, prompt):
        response = self.writer_llm.chat(
            [{"role": "system", "content": WRITER_PROMPT}, {"role": "user", "content": prompt}]
        )
        return (response.content or "").strip()

    def _review(self, task, draft):
        response = self.critic_llm.chat(
            [
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": f"Задание: {task}\n\nТекст:\n{draft}"},
            ]
        )
        return (response.content or "").strip()

    def run(self, task):
        draft = self._write(task)
        for round_number in range(1, self.max_rounds + 1):
            review = self._review(task, draft)
            if review.upper().startswith(APPROVED):
                return {"text": draft, "rounds": round_number, "approved": True}
            if round_number == self.max_rounds:
                break
            draft = self._write(
                f"Задание: {task}\n\nЧерновик:\n{draft}\n\nЗамечания рецензента:\n{review}\n\nИсправь текст."
            )
        return {"text": draft, "rounds": self.max_rounds, "approved": False}

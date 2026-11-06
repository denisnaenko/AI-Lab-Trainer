import solution as sol
from agentkit import ScriptedLLM

ROUTES = {
    "billing": "оплата, счета и возвраты",
    "tech": "технические проблемы с сайтом и приложением",
    "study": "вопросы по учебному плану и расписанию",
}
ANSWERS = {
    "billing": "Счёт отправлен на почту.",
    "tech": "Перезагрузите приложение.",
    "study": "Расписание опубликовано в личном кабинете.",
}


def make_agents(log):
    def make(name):
        def agent(request):
            log.append((name, request))
            return ANSWERS[name]

        return agent

    return {name: make(name) for name in ROUTES}


def make_router(responses):
    llm = ScriptedLLM(responses)
    return sol.Router(llm, ROUTES), llm


def make_system(router_responses, agents=None, **kwargs):
    log = []
    router, llm = make_router(router_responses)
    system = sol.MultiAgentSystem(router, make_agents(log) if agents is None else agents, **kwargs)
    return system, llm, log


def test_r1_route_exact_name():
    router, _ = make_router(["billing"])
    assert router.route("Верните деньги за курс") == "billing"


def test_r1_prompt_lists_routes():
    router, llm = make_router(["study"])
    router.route("Когда экзамен?")
    assert llm.call_count == 1
    messages = llm.messages_of(0)
    assert messages[0]["role"] == "system"
    prompt = messages[-1]["content"]
    for name, description in ROUTES.items():
        assert name in prompt and description in prompt, f"в запросе к модели нет маршрута {name}"
    assert "Когда экзамен?" in prompt


def test_r1_case_and_spaces():
    router, _ = make_router(["  Tech\n"])
    assert router.route("Сайт не открывается") == "tech"


def test_r2_handle_dispatches():
    system, _, _ = make_system(["billing"])
    assert system.handle("Где мой счёт?") == {"route": "billing", "answer": ANSWERS["billing"]}


def test_r2_other_route():
    system, _, _ = make_system(["study"])
    assert system.handle("Где расписание?") == {"route": "study", "answer": ANSWERS["study"]}


def test_r3_single_agent_called_with_request():
    system, _, log = make_system(["tech", "billing"])
    system.handle("Приложение падает")
    system.handle("Нужен возврат")
    assert log == [("tech", "Приложение падает"), ("billing", "Нужен возврат")]


def test_r4_extra_text():
    router, _ = make_router(["Маршрут: Billing."])
    assert router.route("Верните деньги") == "billing"


def test_r4_ambiguous_or_unknown():
    router, _ = make_router(["billing или tech", "не знаю"])
    assert router.route("Сайт списал деньги дважды") is None, "неоднозначный ответ — маршрут не определён"
    assert router.route("Расскажи анекдот") is None


def test_r4_word_boundaries():
    router, _ = make_router(["technical"])
    assert router.route("что-то") is None, "название маршрута должно совпадать со словом целиком"


def test_r5_fallback_message():
    system, _, log = make_system(["не знаю"])
    assert system.handle("Расскажи анекдот") == {"route": sol.FALLBACK_ROUTE, "answer": sol.FALLBACK_MESSAGE}
    assert log == []


def test_r5_fallback_function():
    system, _, _ = make_system(["???"], fallback=lambda request: "Оператор: " + request)
    assert system.handle("Помогите")["answer"] == "Оператор: Помогите"


def test_r5_route_without_agent():
    log = []
    system, _, _ = make_system(["tech"], agents={"billing": make_agents(log)["billing"]})
    assert system.handle("Сайт не работает")["route"] == sol.FALLBACK_ROUTE


def test_r6_log_records():
    system, _, _ = make_system(["billing", "не знаю"])
    system.handle("Нужен счёт")
    system.handle("Расскажи анекдот")
    assert system.log == [
        {"request": "Нужен счёт", "route": "billing", "status": "ok"},
        {"request": "Расскажи анекдот", "route": sol.FALLBACK_ROUTE, "status": "fallback"},
    ]


def failing_agent(request):
    raise RuntimeError("база платежей недоступна")


def test_r7_agent_error_isolated():
    system, _, _ = make_system(["billing"], agents={"billing": failing_agent})
    assert system.handle("Нужен счёт") == {"route": "billing", "answer": sol.AGENT_ERROR_MESSAGE}
    assert system.log[-1]["status"].startswith("error")


def test_r7_system_keeps_working():
    log = []
    agents = make_agents(log)
    agents["billing"] = failing_agent
    system, _, _ = make_system(["billing", "tech"], agents=agents)
    system.handle("Нужен счёт")
    assert system.handle("Сайт не работает") == {"route": "tech", "answer": ANSWERS["tech"]}


def test_r8_writer_critic_approved():
    writer = ScriptedLLM(["черновик 1", "черновик 2"])
    critic = ScriptedLLM(["Добавь пример.", "ОДОБРЕНО"])
    result = sol.WriterCriticLoop(writer, critic).run("Объясни, что такое агент")
    assert result == {"text": "черновик 2", "rounds": 2, "approved": True}
    revision_request = writer.messages_of(1)[-1]["content"]
    assert "черновик 1" in revision_request and "Добавь пример." in revision_request


def test_r8_not_approved_limit():
    writer = ScriptedLLM(["в1", "в2"])
    critic = ScriptedLLM(["плохо", "всё ещё плохо"])
    result = sol.WriterCriticLoop(writer, critic, max_rounds=2).run("Тема")
    assert result == {"text": "в2", "rounds": 2, "approved": False}
    assert writer.call_count == 2 and critic.call_count == 2


def test_r8_critic_sees_task_and_draft():
    writer = ScriptedLLM(["черновик"])
    critic = ScriptedLLM(["ОДОБРЕНО: отлично"])
    result = sol.WriterCriticLoop(writer, critic).run("Объясни RAG")
    assert result == {"text": "черновик", "rounds": 1, "approved": True}
    review_request = critic.messages_of(0)[-1]["content"]
    assert "Объясни RAG" in review_request and "черновик" in review_request
    assert "Объясни RAG" in writer.messages_of(0)[-1]["content"]

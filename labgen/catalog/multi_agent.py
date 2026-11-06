from .base import MutantDef, RequirementDef, TopicDef

TOPIC = TopicDef(
    id="multi_agent",
    title="Мультиагентная система",
    summary="Диспетчер на LLM, маршрутизация к исполнителям, запасной маршрут, изоляция сбоев, цикл «автор — критик».",
    keywords=(
        "мультиагент", "мультиагентная", "мультиагентные", "несколько агентов", "multi-agent", "multiagent",
        "маршрутизация", "роутер", "router", "оркестрация", "оркестратор", "критик", "рецензент",
        "автор и критик", "диспетчер", "супервизор",
    ),
    context_md="""
Сложные задачи удобно делить между несколькими специализированными агентами. Типичная схема:
диспетчер (router) на основе LLM определяет, к какому исполнителю относится запрос, а исполнители
решают свою узкую задачу. Другой распространённый приём — пара «автор — критик»: один агент
пишет текст, второй его рецензирует, и автор исправляет текст по замечаниям.

В работе вы реализуете службу поддержки студентов с диспетчером и исполнителями, а на
продвинутом уровне — ещё и цикл «автор — критик».
""",
    interface_md={
        1: """
Реализуйте в `solution.py` класс `Router(llm, routes)` с методом `route(request)` и класс
`MultiAgentSystem(router, agents)` с методом `handle(request)`. `routes` — словарь
`{название: описание}`, `agents` — словарь `{название: функция-исполнитель}`.
""",
        2: """
Реализуйте `Router` и `MultiAgentSystem(router, agents, fallback=None)` с методом `handle` и
журналом `log`. Модель-диспетчер не всегда отвечает строго по формату, а некоторые запросы не
подходят ни одному исполнителю.
""",
        3: """
Реализуйте `Router`, `MultiAgentSystem(router, agents, fallback=None)` и
`WriterCriticLoop(writer_llm, critic_llm, max_rounds=3)` с методом `run(task)`.
Исполнители могут падать с исключениями — система должна продолжать работу.
""",
    },
    requirements=[
        RequirementDef("R1", 1, "`Router.route(request)` одним вызовом модели передаёт системную инструкцию и сообщение со всеми маршрутами (название и описание) и текстом запроса; возвращает название маршрута из ответа модели без учёта регистра и пробелов по краям."),
        RequirementDef("R2", 1, "`MultiAgentSystem.handle(request)` определяет маршрут через роутер, передаёт запрос исполнителю этого маршрута и возвращает `{\"route\": маршрут, \"answer\": ответ исполнителя}`."),
        RequirementDef("R3", 1, "`handle` вызывает ровно одного исполнителя и передаёт ему исходный текст запроса без изменений."),
        RequirementDef("R4", 2, "Ответ диспетчера может содержать лишний текст и знаки препинания («Маршрут: Billing.»): маршрут определяется, если в ответе ровно одно название маршрута отдельным словом; если названий нет или их несколько, `route` возвращает `None`."),
        RequirementDef("R5", 2, "Если маршрут не определён или для него нет исполнителя, `handle` возвращает `{\"route\": FALLBACK_ROUTE, \"answer\": ...}` — ответ функции `fallback(request)`, если она задана, иначе `FALLBACK_MESSAGE`."),
        RequirementDef("R6", 2, "Атрибут `log` содержит запись `{\"request\", \"route\", \"status\"}` для каждого обработанного запроса; `status` — `\"ok\"` или `\"fallback\"`."),
        RequirementDef("R7", 3, "Если исполнитель выбросил исключение, `handle` возвращает `{\"route\": маршрут, \"answer\": AGENT_ERROR_MESSAGE}`, в журнале `status` начинается с `\"error\"`, а система продолжает обрабатывать следующие запросы."),
        RequirementDef("R8", 3, "`WriterCriticLoop.run(task)`: автор пишет черновик, критик его проверяет; ответ критика, начинающийся с `APPROVED`, завершает цикл; иначе автор получает черновик и замечания и переписывает текст. Рецензий не больше `max_rounds`; результат — `{\"text\", \"rounds\", \"approved\"}`."),
    ],
    public_tests={
        "test_r1_route_exact_name",
        "test_r2_handle_dispatches",
        "test_r3_single_agent_called_with_request",
        "test_r4_extra_text",
        "test_r5_fallback_message",
        "test_r6_log_records",
        "test_r7_agent_error_isolated",
        "test_r8_writer_critic_approved",
    },
    mutants=[
        MutantDef("M1", 1, "Маршруты не перечислены в запросе к модели",
                  'f"Маршруты:\\n{lines}\\n\\nЗапрос: {request}"', 'f"Запрос: {request}"'),
        MutantDef("M2", 1, "Ответ диспетчера не нормализуется",
                  'answer = (response.content or "").strip().lower()', 'answer = response.content or ""'),
        MutantDef("M3", 1, "Исполнитель получает пустой запрос",
                  "answer = self.agents[route](request)", 'answer = self.agents[route]("")'),
        MutantDef("M4", 1, "handle возвращает маршрут вместо ответа",
                  'return {"route": route, "answer": answer}', 'return {"route": route, "answer": route}'),
        MutantDef("M5", 2, "Неоднозначный ответ принимается", "        if len(found) == 1:", "        if found:"),
        MutantDef("M6", 2, "Поиск названия по подстроке, а не по слову",
                  "if key in words]", "if key in answer]"),
        MutantDef("M7", 2, "Нет запасного маршрута", "        if route not in self.agents:", "        if False:"),
        MutantDef("M8", 2, "Журнал не ведётся",
                  '        self.log.append({"request": request, "route": route, "status": status})', "        pass"),
        MutantDef("M9", 3, "Исключение исполнителя не перехватывается",
                  "            except Exception as exc:", "            except KeyError as exc:"),
        MutantDef("M10", 3, "Замечания критика не передаются автору",
                  'f"Задание: {task}\\n\\nЧерновик:\\n{draft}\\n\\nЗамечания рецензента:\\n{review}\\n\\nИсправь текст."',
                  'f"Задание: {task}\\n\\nИсправь текст."'),
        MutantDef("M11", 3, "Цикл не останавливается после одобрения",
                  "            if review.upper().startswith(APPROVED):", "            if False:"),
        MutantDef("M12", 3, "Лишний раунд автора после последней рецензии",
                  "            if round_number == self.max_rounds:\n                break\n", ""),
    ],
    hints={
        1: [
            "Чем понятнее описаны маршруты в запросе, тем точнее модель выбирает исполнителя.",
            "Ответ модели перед сравнением стоит очистить: `strip()` и `lower()`.",
        ],
        2: [
            "Слова ответа удобно получить через `re.findall(r\"\\w+\", text)`.",
            "Запасной маршрут — нормальная часть системы, а не исключительная ситуация.",
        ],
        3: [
            "Сбой одного исполнителя не должен останавливать всю систему — это и есть изоляция ошибок.",
            "Посчитайте, сколько раз вызываются автор и критик при `max_rounds=2`, если критик ни разу не одобрил текст.",
        ],
    },
    minutes={1: 45, 2: 90, 3: 140},
)

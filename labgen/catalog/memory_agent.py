from .base import MutantDef, RequirementDef, TopicDef

TOPIC = TopicDef(
    id="memory_agent",
    title="Агент с памятью диалога",
    summary="История диалога, скользящее окно контекста, атомарность обновления памяти, сжатие истории через LLM.",
    keywords=(
        "память", "памятью", "memory", "история диалога", "истории диалога", "окно контекста",
        "чат-бот", "чат-бота", "summary", "суммаризация", "сжатие истории", "conversation",
    ),
    context_md="""
Модель не помнит прошлых сообщений: всё, что она «знает» о диалоге, агент должен каждый раз
передать ей в списке `messages`. Поэтому память чат-агента — это код, который хранит историю,
решает, какую её часть отправить модели, и не даёт контексту бесконечно расти.

В этой работе вы реализуете такую память: от простой истории до скользящего окна и сжатия
старых реплик в краткое содержание с помощью самой модели.
""",
    interface_md={
        1: """
Реализуйте класс `ChatAgent` в `solution.py`: метод `send(user_message)` отправляет сообщение
модели с учётом истории и возвращает ответ, метод `reset()` начинает диалог заново, атрибут
`history` хранит реплики диалога.
""",
        2: """
Реализуйте класс `ChatAgent(llm, system_prompt=..., max_history=None)` с методами `send` и `reset`
и атрибутом `history`. Длина истории ограничивается окном `max_history`, а сбой модели не должен
портить историю.
""",
        3: """
Реализуйте класс `ChatAgent(llm, system_prompt=..., max_history=None, summarize_threshold=None)`
с методами `send` и `reset` и атрибутами `history` и `summary`. Когда история становится длинной,
агент просит модель кратко пересказать старую часть диалога и дальше работает с этим пересказом.
""",
    },
    requirements=[
        RequirementDef("R1", 1, "`send(text)` отправляет модели системное сообщение (`system_prompt`), предыдущую историю и новое сообщение пользователя и возвращает текст ответа."),
        RequirementDef("R2", 1, "`history` хранит сообщения пользователя и ответы модели по порядку (без системного сообщения); следующий вызов `send` передаёт модели прошлые реплики."),
        RequirementDef("R3", 1, "`reset()` очищает историю."),
        RequirementDef("R4", 2, "Если задан `max_history`, в истории остаются только последние `max_history` сообщений; системное сообщение отправляется всегда. Без `max_history` история не ограничена."),
        RequirementDef("R5", 2, "После обрезки история начинается с сообщения пользователя: ответ ассистента без вопроса отбрасывается."),
        RequirementDef("R6", 2, "Если вызов модели завершился исключением, история не меняется, а исключение передаётся вызывающему коду."),
        RequirementDef("R7", 3, "Если задан `summarize_threshold` и история длиннее порога, перед ответом агент отдельным вызовом модели (системное сообщение `SUMMARY_PROMPT` и текст старых реплик) получает краткое содержание и сохраняет его в `summary`; в истории остаются последние `KEEP_LAST` сообщений."),
        RequirementDef("R8", 3, "Краткое содержание передаётся модели вторым системным сообщением `SUMMARY_PREFIX + summary`; при следующем сжатии старое резюме включается в текст для пересказа; `reset()` очищает и резюме."),
    ],
    public_tests={
        "test_r1_first_message",
        "test_r2_history_sent_on_next_call",
        "test_r3_reset",
        "test_r4_window",
        "test_r5_window_starts_with_user",
        "test_r6_history_unchanged_on_error",
        "test_r7_summary_call",
        "test_r8_summary_in_context",
    },
    mutants=[
        MutantDef("M1", 1, "Системное сообщение не отправляется",
                  '        messages = [{"role": "system", "content": self.system_prompt}]', "        messages = []"),
        MutantDef("M2", 1, "Ответ модели не сохраняется в истории",
                  '        self.history = self.history + [user, {"role": "assistant", "content": reply}]',
                  "        self.history = self.history + [user]"),
        MutantDef("M3", 1, "reset() не очищает историю",
                  "    def reset(self):\n        \"\"\"Начать диалог заново.\"\"\"\n        self.history = []\n",
                  "    def reset(self):\n        \"\"\"Начать диалог заново.\"\"\"\n"),
        MutantDef("M4", 2, "История не обрезается",
                  "        trimmed = self.history[-self.max_history:]", "        trimmed = list(self.history)"),
        MutantDef("M5", 2, "Ответ ассистента без вопроса не отбрасывается",
                  '        while trimmed and trimmed[0]["role"] != "user":', "        while False:"),
        MutantDef("M6", 2, "История меняется до успешного ответа модели",
                  '        response = self.llm.chat(self._context() + [user])\n        reply = response.content or ""\n'
                  '        self.history = self.history + [user, {"role": "assistant", "content": reply}]',
                  '        self.history = self.history + [user]\n        response = self.llm.chat(self._context())\n'
                  '        reply = response.content or ""\n'
                  '        self.history = self.history + [{"role": "assistant", "content": reply}]'),
        MutantDef("M7", 3, "История не сжимается",
                  "        if self.summarize_threshold is not None and len(self.history) > self.summarize_threshold:",
                  "        if False:"),
        MutantDef("M8", 3, "Резюме не передаётся модели",
                  "        if self.summary:\n            messages.append(", "        if False:\n            messages.append("),
        MutantDef("M9", 3, "Старое резюме теряется при новом сжатии",
                  "        if self.summary:\n            lines.append(", "        if False:\n            lines.append("),
        MutantDef("M10", 3, "Сжатие начинается раньше порога",
                  "len(self.history) > self.summarize_threshold", "len(self.history) >= self.summarize_threshold"),
    ],
    hints={
        1: [
            "Собирайте список для модели заново при каждом вызове: системное сообщение + история + новое сообщение.",
            "Сообщения ассистента в истории имеют роль `assistant`.",
        ],
        2: [
            "Срез `history[-n:]` возвращает последние n элементов.",
            "Историю безопаснее обновлять только после успешного ответа модели.",
        ],
        3: [
            "Сжатие — это отдельный вызов модели со своей инструкцией, не смешивайте его с основным диалогом.",
            "Подумайте, что произойдёт с резюме после второго и третьего сжатия.",
        ],
    },
    minutes={1: 40, 2: 80, 3: 130},
)

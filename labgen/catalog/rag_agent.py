from .base import MutantDef, RequirementDef, TopicDef

TOPIC = TopicDef(
    id="rag_agent",
    title="RAG: ответы по базе знаний",
    summary="Поиск документов, сборка контекста, ссылки на источники, отказ без контекста, переформулировка запроса.",
    keywords=(
        "rag", "раг", "база знаний", "базе знаний", "поиск", "retrieval", "документ", "документы",
        "источник", "источники", "retrieval augmented", "поиск по документам", "векторный",
    ),
    context_md="""
Модель знает только то, на чём её обучили, и легко выдумывает факты о конкретном курсе.
Подход RAG (Retrieval-Augmented Generation) решает это так: агент сначала находит в базе знаний
подходящие документы, передаёт их модели как контекст и просит отвечать только по ним,
со ссылками на источники.

В работе используется простейший поиск по совпадению слов — его достаточно, чтобы разобраться
в устройстве RAG-агента. В реальных системах вместо него используют векторный поиск.
""",
    interface_md={
        1: """
Реализуйте в `solution.py` функцию `tokenize(text)`, класс `KnowledgeBase` с методом
`search(query, top_k=3)` и класс `RAGAgent` с методом `answer(question)`, который возвращает
словарь `{"answer": ..., "sources": [...]}`.
""",
        2: """
Реализуйте `tokenize`, `KnowledgeBase` (методы `add` и `search`) и `RAGAgent.answer`.
Поиск должен игнорировать служебные слова из `STOPWORDS`, а агент — не обращаться к модели,
если подходящих документов нет.
""",
        3: """
Реализуйте `tokenize`, `KnowledgeBase` (`add`, `search`) и `RAGAgent(llm, knowledge_base, top_k=3,
rephrase=False)` с методом `answer`. Источниками считаются документы, на которые модель сослалась,
а если поиск ничего не нашёл, агент может попросить модель переформулировать вопрос.
""",
    },
    requirements=[
        RequirementDef("R1", 1, "`tokenize(text)` возвращает слова текста в нижнем регистре без знаков препинания."),
        RequirementDef("R2", 1, "`KnowledgeBase.search(query, top_k)` считает число общих слов запроса с заголовком и текстом документа, отбрасывает документы без совпадений и возвращает до `top_k` документов по убыванию совпадений (при равенстве — в исходном порядке)."),
        RequirementDef("R3", 1, "`RAGAgent.answer(question)` одним вызовом модели передаёт системную инструкцию и сообщение с найденными документами (каждый с меткой `[id]`) и вопросом; возвращает `{\"answer\": ответ модели, \"sources\": [id найденных документов]}`."),
        RequirementDef("R4", 2, "Если подходящих документов нет, `answer` возвращает `{\"answer\": NO_ANSWER, \"sources\": []}` и не вызывает модель."),
        RequirementDef("R5", 2, "`tokenize` отбрасывает слова из `STOPWORDS`, поэтому служебные слова не дают совпадений при поиске."),
        RequirementDef("R6", 2, "`KnowledgeBase.add(document)` добавляет документ, и он сразу участвует в поиске; документ с уже существующим `id` (в том числе при создании базы) приводит к `ValueError`."),
        RequirementDef("R7", 3, "`sources` содержит только найденные документы, на которые модель сослалась в ответе в виде `[id]`; неизвестные ссылки игнорируются; если ссылок нет, возвращаются все найденные документы."),
        RequirementDef("R8", 3, "При `rephrase=True`, если поиск ничего не нашёл, агент один раз просит модель переформулировать вопрос (системное сообщение `REPHRASE_PROMPT` и исходный вопрос) и ищет снова; если и теперь документов нет — возвращает `NO_ANSWER`."),
    ],
    public_tests={
        "test_r1_tokenize_lowercase_words",
        "test_r2_search_ranks_by_overlap",
        "test_r3_answer_with_context",
        "test_r4_no_documents_no_llm_call",
        "test_r5_stopwords_removed",
        "test_r6_add_document",
        "test_r7_only_cited_sources",
        "test_r8_rephrase_when_nothing_found",
    },
    mutants=[
        MutantDef("M1", 1, "Нет приведения к нижнему регистру",
                  're.findall(r"\\w+", text.lower())', 're.findall(r"\\w+", text)'),
        MutantDef("M2", 1, "Документы не сортируются по релевантности",
                  "        scored.sort(key=lambda item: (-item[0], item[1]))", "        pass"),
        MutantDef("M3", 1, "Контекст не попадает в запрос",
                  'f"Контекст:\\n{context}\\n\\nВопрос: {question}"', 'f"Вопрос: {question}"'),
        MutantDef("M4", 1, "Документы без совпадений попадают в выдачу", "            if score > 0:", "            if score >= 0:"),
        MutantDef("M5", 1, "Игнорируется top_k",
                  "return [doc for _, _, doc in scored[:top_k]]", "return [doc for _, _, doc in scored]"),
        MutantDef("M6", 2, "Стоп-слова не отбрасываются",
                  '[w for w in re.findall(r"\\w+", text.lower()) if w not in STOPWORDS]',
                  're.findall(r"\\w+", text.lower())'),
        MutantDef("M7", 2, "Модель вызывается без найденных документов",
                  '        if not documents:\n            return {"answer": NO_ANSWER, "sources": []}\n', ""),
        MutantDef("M8", 2, "Повторяющиеся id не проверяются",
                  '        if any(d["id"] == document["id"] for d in self.documents):', "        if False:"),
        MutantDef("M9", 3, "Источники не фильтруются по ссылкам", "        return cited or found", "        return found"),
        MutantDef("M10", 3, "Нет переформулировки вопроса", "        if documents or not self.rephrase:", "        if True:"),
    ],
    hints={
        1: [
            "Для подсчёта общих слов удобны множества: `len(set_a & set_b)`.",
            "Сортировка по ключу `(-score, index)` даёт убывание совпадений и сохраняет исходный порядок при равенстве.",
        ],
        2: [
            "Пустой вопрос — тоже случай «документов нет».",
            "Токены документов можно посчитать один раз при добавлении, а не при каждом поиске.",
        ],
        3: [
            "Ссылку на документ удобно искать как подстроку `[id]` в ответе модели.",
            "Переформулировка — отдельный вызов модели со своей инструкцией.",
        ],
    },
    minutes={1: 50, 2: 90, 3: 135},
)

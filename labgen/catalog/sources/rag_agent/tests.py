import solution as sol
from agentkit import ScriptedLLM

DOCS = [
    {"id": "d1", "title": "Дедлайны", "text": "Лабораторные работы сдаются до пятницы через Moodle."},
    {"id": "d2", "title": "Оценивание", "text": "Итоговая оценка складывается из лабораторных работ и экзамена."},
    {"id": "d3", "title": "Консультации", "text": "Консультации проходят по средам в аудитории 305."},
    {"id": "d4", "title": "Экзамен", "text": "Экзамен проходит письменно, допускаются конспекты."},
]


def make_agent(responses, **kwargs):
    llm = ScriptedLLM(responses)
    return sol.RAGAgent(llm, sol.KnowledgeBase(DOCS), **kwargs), llm


def ids(documents):
    return [d["id"] for d in documents]


def raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"ожидалось исключение {exc_type.__name__}")


def test_r1_tokenize_lowercase_words():
    assert sol.tokenize("Привет, Мир! Agents 2024") == ["привет", "мир", "agents", "2024"]


def test_r1_tokenize_punctuation():
    assert sol.tokenize("Moodle: сдача — пятница!!!") == ["moodle", "сдача", "пятница"]


def test_r2_search_ranks_by_overlap():
    kb = sol.KnowledgeBase(DOCS)
    assert ids(kb.search("проходит консультации экзамен")) == ["d4", "d3"]


def test_r2_top_k_and_ties():
    kb = sol.KnowledgeBase(DOCS)
    assert ids(kb.search("экзамен консультации")) == ["d3", "d4"], "при равенстве — исходный порядок"
    assert ids(kb.search("экзамен консультации", top_k=1)) == ["d3"]


def test_r2_no_zero_score():
    kb = sol.KnowledgeBase(DOCS)
    assert kb.search("квантовая физика") == []


def test_r3_answer_with_context():
    agent, llm = make_agent(["Консультации по средам."])
    result = agent.answer("консультации аудитории")
    assert result == {"answer": "Консультации по средам.", "sources": ["d3"]}
    prompt = llm.messages_of(0)[-1]["content"]
    assert "[d3]" in prompt and "аудитории 305" in prompt, "в запрос нужно включить найденные документы"
    assert "консультации аудитории" in prompt, "в запрос нужно включить вопрос"


def test_r3_single_call_and_top_k():
    agent, llm = make_agent(["ответ"], top_k=1)
    result = agent.answer("экзамен консультации")
    assert llm.call_count == 1
    assert llm.messages_of(0)[0]["role"] == "system"
    assert result["sources"] == ["d3"]
    assert "[d4]" not in llm.messages_of(0)[-1]["content"]


def test_r4_no_documents_no_llm_call():
    agent, llm = make_agent([])
    assert agent.answer("квантовая физика") == {"answer": sol.NO_ANSWER, "sources": []}
    assert llm.call_count == 0, "если документов нет, модель вызывать не нужно"


def test_r4_empty_question():
    agent, llm = make_agent([])
    assert agent.answer("") == {"answer": sol.NO_ANSWER, "sources": []}
    assert llm.call_count == 0


def test_r5_stopwords_removed():
    assert sol.tokenize("Что такое агент и как он работает") == ["такое", "агент", "работает"]


def test_r5_stopwords_do_not_match():
    kb = sol.KnowledgeBase(DOCS)
    assert kb.search("как и где") == []
    assert ids(kb.search("что в экзамен")) == ["d4"], "стоп-слова не должны давать совпадений"


def test_r6_add_document():
    kb = sol.KnowledgeBase(DOCS)
    kb.add({"id": "d5", "title": "Библиотека", "text": "Библиотека работает до восьми вечера."})
    assert ids(kb.search("библиотека")) == ["d5"]


def test_r6_duplicate_id():
    kb = sol.KnowledgeBase(DOCS)
    raises(ValueError, kb.add, {"id": "d1", "title": "Копия", "text": "Другой текст"})
    raises(ValueError, sol.KnowledgeBase, [DOCS[0], DOCS[0]])


def test_r7_only_cited_sources():
    agent, _ = make_agent(["Оценка зависит от лабораторных [d2]."])
    assert agent.answer("экзамен оценка лабораторных")["sources"] == ["d2"]


def test_r7_unknown_citations_ignored():
    agent, _ = make_agent(["См. [d2] и [d9].", "Без ссылок."])
    assert agent.answer("экзамен оценка лабораторных")["sources"] == ["d2"]
    assert agent.answer("экзамен оценка лабораторных")["sources"] == ["d2", "d4"]


def test_r8_rephrase_when_nothing_found():
    agent, llm = make_agent(["консультации аудитория", "По средам [d3]."], rephrase=True)
    result = agent.answer("где помощь преподавателя")
    assert result == {"answer": "По средам [d3].", "sources": ["d3"]}
    assert llm.call_count == 2
    assert llm.messages_of(0) == [
        {"role": "system", "content": sol.REPHRASE_PROMPT},
        {"role": "user", "content": "где помощь преподавателя"},
    ]


def test_r8_no_rephrase_when_found():
    agent, llm = make_agent(["ответ"], rephrase=True)
    agent.answer("консультации аудитории")
    assert llm.call_count == 1


def test_r8_still_nothing_after_rephrase():
    agent, llm = make_agent(["квантовая механика"], rephrase=True)
    assert agent.answer("где помощь преподавателя") == {"answer": sol.NO_ANSWER, "sources": []}
    assert llm.call_count == 1

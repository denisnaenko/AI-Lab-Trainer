"""Качество каталога: все 15 работ проходят проверку, тесты ловят типичные ошибки,
а минимальное решение базового уровня не требует знаний старших уровней."""

import pytest

from labgen.catalog import TOPICS, build_lab, match_topic
from labgen.grader import grade_solution
from labgen.models import requirement_of, test_names
from labgen.pipeline.validator import validate_spec

LEVELS = [1, 2, 3]


@pytest.mark.slow
@pytest.mark.parametrize("topic_id", list(TOPICS))
@pytest.mark.parametrize("level", LEVELS)
def test_catalog_lab_passes_validation(topic_id, level):
    spec = build_lab(topic_id, level)
    report = validate_spec(spec)
    assert report.passed, [f"{c.name}: {c.details}" for c in report.errors()]
    assert report.auto_coverage >= 0.8
    assert report.mutation_score == 1.0
    assert report.starter_fail_rate == 1.0
    assert spec.total_points == 100


@pytest.mark.parametrize("topic_id", list(TOPICS))
@pytest.mark.parametrize("level", LEVELS)
def test_every_requirement_has_public_test(topic_id, level):
    spec = build_lab(topic_id, level)
    public = {requirement_of(n) for n in test_names(spec.tests_public)}
    for req in spec.requirements:
        if req.auto:
            assert req.id in public, f"{topic_id} L{level}: нет публичного теста для {req.id}"


@pytest.mark.parametrize("topic_id", list(TOPICS))
def test_levels_grow(topic_id):
    counts = [len([r for r in build_lab(topic_id, lvl).requirements if r.auto]) for lvl in LEVELS]
    assert counts == sorted(counts) and counts[0] < counts[-1]


MINIMAL_L1 = {
    "structured_output": '''
import json
SYSTEM_PROMPT = "Извлеки товар, количество и город. Ответь только JSON с полями product, quantity, city."
def build_messages(text):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
def parse_response(content):
    return json.loads(content)
def extract_order(llm, text):
    return parse_response(llm.chat(build_messages(text)).content)
''',
    "tool_agent": '''
from agentkit import assistant_message, tool_result_message
DEFAULT_SYSTEM_PROMPT = "Ты помощник."
class ToolAgent:
    def __init__(self, llm, tools, tool_schemas, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.llm, self.tools, self.tool_schemas, self.system_prompt = llm, tools, tool_schemas, system_prompt
    def run(self, user_message):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]
        response = self.llm.chat(messages, tools=self.tool_schemas)
        if not response.tool_calls:
            return response.content
        messages.append(assistant_message(response))
        for call in response.tool_calls:
            messages.append(tool_result_message(call, self.tools[call.name](**call.arguments)))
        return self.llm.chat(messages, tools=self.tool_schemas).content
''',
    "memory_agent": '''
DEFAULT_SYSTEM_PROMPT = "Ты — бот."
class ChatAgent:
    def __init__(self, llm, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.llm, self.system_prompt, self.history = llm, system_prompt, []
    def send(self, user_message):
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        reply = self.llm.chat(messages + [{"role": "user", "content": user_message}]).content
        self.history += [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        return reply
    def reset(self):
        self.history = []
''',
    "rag_agent": '''
import re
RAG_SYSTEM_PROMPT = "Отвечай по контексту."
def tokenize(text):
    return re.findall(r"\\w+", text.lower())
class KnowledgeBase:
    def __init__(self, documents):
        self.documents = list(documents)
    def search(self, query, top_k=3):
        q = set(tokenize(query))
        scored = []
        for i, d in enumerate(self.documents):
            s = len(q & set(tokenize(d["title"] + " " + d["text"])))
            if s > 0:
                scored.append((-s, i, d))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [d for _, _, d in scored[:top_k]]
class RAGAgent:
    def __init__(self, llm, knowledge_base, top_k=3):
        self.llm, self.kb, self.top_k = llm, knowledge_base, top_k
    def answer(self, question):
        docs = self.kb.search(question, self.top_k)
        context = "\\n".join(f"[{d['id']}] {d['title']}: {d['text']}" for d in docs)
        r = self.llm.chat([{"role": "system", "content": RAG_SYSTEM_PROMPT},
                           {"role": "user", "content": f"Контекст:\\n{context}\\nВопрос: {question}"}])
        return {"answer": r.content, "sources": [d["id"] for d in docs]}
''',
    "multi_agent": '''
ROUTER_PROMPT = "Выбери маршрут."
class Router:
    def __init__(self, llm, routes):
        self.llm, self.routes = llm, routes
    def route(self, request):
        lines = "\\n".join(f"- {n}: {d}" for n, d in self.routes.items())
        r = self.llm.chat([{"role": "system", "content": ROUTER_PROMPT},
                           {"role": "user", "content": f"{lines}\\nЗапрос: {request}"}])
        return r.content.strip().lower()
class MultiAgentSystem:
    def __init__(self, router, agents):
        self.router, self.agents = router, agents
    def handle(self, request):
        route = self.router.route(request)
        return {"route": route, "answer": self.agents[route](request)}
''',
}


@pytest.mark.parametrize("topic_id", list(TOPICS))
def test_minimal_level1_solution_gets_full_auto_score(topic_id):
    spec = build_lab(topic_id, 1)
    report = grade_solution(spec, MINIMAL_L1[topic_id])
    assert report.auto_score == report.auto_max, report.feedback


@pytest.mark.parametrize("topic_id", list(TOPICS))
def test_minimal_level1_solution_is_not_enough_for_level3(topic_id):
    spec = build_lab(topic_id, 3)
    report = grade_solution(spec, MINIMAL_L1[topic_id])
    assert report.auto_score < report.auto_max * 0.75


@pytest.mark.parametrize(
    "request_text, expected",
    [
        ("Создание агента с инструментами", "tool_agent"),
        ("Function calling", "tool_agent"),
        ("Память чат-бота и суммаризация", "memory_agent"),
        ("Агент с памятью", "memory_agent"),
        ("RAG по базе знаний курса", "rag_agent"),
        ("Оркестрация нескольких агентов", "multi_agent"),
        ("Извлечение JSON из ответа модели", "structured_output"),
    ],
)
def test_topic_matching(request_text, expected):
    assert match_topic(request_text).id == expected


@pytest.mark.parametrize("request_text", ["Квантовая химия", "История Древнего Рима", "Алгоритмы поиска"])
def test_unknown_topic(request_text):
    assert match_topic(request_text) is None

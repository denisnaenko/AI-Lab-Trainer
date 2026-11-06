"""Агент на LangGraph: маршруты графа, цикл исправления, рецензент, симуляция студентов."""

import json

import pytest

from labgen.agent import LAB_AGENT, generate_lab
from labgen.catalog import build_lab
from labgen.llm import LLMError, fake_llm
from labgen.pipeline.generator import TopicNotFound
from labgen.pipeline.jsonutil import extract_json_object
from labgen.pipeline.reviewer import heuristic_review
from labgen.pipeline.students import simulate_students


def spec_json(topic_id="memory_agent", level=2, **overrides):
    data = build_lab(topic_id, level).model_dump(exclude={"topic_id", "level"})
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_graph_structure():
    graph = LAB_AGENT.get_graph()
    nodes = set(graph.nodes)
    assert {"plan", "catalog", "generate", "validate", "finalize", "review", "simulate"} <= nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("validate", "generate") in edges, "цикл исправления по замечаниям валидатора"
    assert ("generate", "generate") in edges, "повтор при неразобранном ответе"


def test_catalog_mode_offline():
    steps = []
    result = generate_lab("Создание агента с инструментами", 1, on_step=lambda node, log: steps.append(node))
    assert result.mode == "catalog" and result.report.passed
    assert result.spec.topic_id == "tool_agent"
    assert steps == ["plan", "catalog", "validate", "review", "simulate"]


def test_catalog_mode_unknown_topic():
    with pytest.raises(TopicNotFound):
        generate_lab("История Древнего Рима", 1, mode="catalog")


def test_llm_mode_requires_model():
    with pytest.raises(LLMError):
        generate_lab("Агент", 1, mode="llm")


def test_llm_mode_repairs_broken_spec():
    good = build_lab("memory_agent", 2)
    broken_reference = good.reference_solution.replace("        self._trim()\n", "")
    llm = fake_llm(
        [
            "Извините, не могу",  # не JSON
            "```json\n" + spec_json(reference_solution=broken_reference) + "\n```",  # эталон не проходит тесты
            spec_json(),  # исправленная работа
            json.dumps({"remarks": [{"severity": "minor", "section": "условие", "comment": "Уточнить пример"}]}),
        ]
    )
    steps = []
    result = generate_lab("Память чат-бота", 2, llm=llm, on_step=lambda node, log: steps.append(node))
    assert result.mode == "llm" and result.report.passed
    assert steps == ["plan", "generate", "generate", "validate", "generate", "validate", "review", "simulate"]
    calls = llm.chat.calls
    assert len(calls) == 4
    assert "Эталон проходит все тесты" in calls[2][1]["content"], "генератор должен получить замечания валидатора"
    assert any(r.comment == "Уточнить пример" for r in result.review)
    assert result.llm_name == "fake"


def test_llm_mode_keeps_best_draft_when_attempts_exhausted():
    broken = build_lab("tool_agent", 1).reference_solution.replace("tools=self.tool_schemas", "tools=None")
    llm = fake_llm([spec_json("tool_agent", 1, reference_solution=broken)] * 2 + ['{"remarks": []}'])
    steps = []
    result = generate_lab("Агент с инструментами", 1, llm=llm, max_attempts=2, on_step=lambda n, l: steps.append(n))
    assert not result.report.passed
    assert "finalize" in steps
    assert any("черновик" in line for line in result.log)


def test_llm_error_without_any_draft():
    llm = fake_llm([RuntimeError("401 Unauthorized")])
    with pytest.raises(LLMError, match="401"):
        generate_lab("Агент", 1, llm=llm)


def test_generator_prompt_contains_level_example_and_materials():
    llm = fake_llm([spec_json("rag_agent", 3), '{"remarks": []}'])
    generate_lab("Поиск по документам", 3, llm=llm, materials="Лекция: BM25 и эмбеддинги")
    system, user = llm.chat.calls[0]
    assert system["role"] == "system" and "ScriptedLLM" in system["content"]
    assert "Продвинутый" in user["content"]
    assert "RAG: ответы по базе знаний" in user["content"], "образец — ближайшая тема каталога"
    assert "BM25" in user["content"]


def test_simulated_students_node():
    spec_text = spec_json("memory_agent", 1)
    reference = build_lab("memory_agent", 1).reference_solution
    llm = fake_llm([spec_text, '{"remarks": []}', "```python\n" + reference + "```"])
    result = generate_lab("Агент с памятью", 1, llm=llm, simulate=1)
    assert result.simulation["solved"] == 1


def test_heuristic_review_flags_requirement_count():
    spec = build_lab("tool_agent", 1)
    spec.level = 3
    assert any("Требований с автопроверкой" in r.comment for r in heuristic_review(spec))


def test_simulate_students_directly():
    spec = build_lab("memory_agent", 1)
    llm = fake_llm(["```python\n" + spec.reference_solution + "```", "```python\n" + spec.starter_code + "```"])
    result = simulate_students(llm, spec, n=2)
    assert result["solved"] == 1 and result["solve_rate"] == 0.5


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Вот:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json_object('ответ {"a": 3} конец') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json_object("нет json")

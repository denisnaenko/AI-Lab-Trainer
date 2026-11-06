import json

import pytest

from labgen.catalog import build_lab
from labgen.llm import ReplayProvider
from labgen.pipeline.generator import TopicNotFound
from labgen.pipeline.jsonutil import extract_json_object
from labgen.pipeline.orchestrator import generate_lab
from labgen.pipeline.reviewer import heuristic_review
from labgen.pipeline.students import simulate_students


def spec_json(topic_id="memory_agent", level=2, **overrides):
    data = build_lab(topic_id, level).model_dump(exclude={"topic_id", "level"})
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_catalog_mode_offline():
    result = generate_lab("Создание агента с инструментами", 1, mode="auto")
    assert result.mode == "catalog"
    assert result.report.passed
    assert result.spec.topic_id == "tool_agent"


def test_catalog_mode_unknown_topic():
    with pytest.raises(TopicNotFound):
        generate_lab("История Древнего Рима", 1, mode="catalog")


def test_llm_mode_repairs_broken_spec():
    good = build_lab("memory_agent", 2)
    broken_reference = good.reference_solution.replace("        self._trim()\n", "")
    provider = ReplayProvider(
        [
            "Извините, не могу",  # не JSON
            "```json\n" + spec_json(reference_solution=broken_reference) + "\n```",  # эталон не проходит тесты
            spec_json(),  # исправленная работа
            json.dumps({"remarks": [{"severity": "minor", "section": "условие", "comment": "Уточнить пример"}]}),
        ]
    )
    result = generate_lab("Память чат-бота", 2, mode="llm", provider=provider)
    assert result.mode == "llm"
    assert result.report.passed
    assert len(provider.calls) == 4
    repair_prompt = provider.calls[2][1]["content"]
    assert "Эталон проходит все тесты" in repair_prompt, "генератор должен получить замечания валидатора"
    assert any(r.comment == "Уточнить пример" for r in result.review)


def test_llm_mode_keeps_best_draft_when_attempts_exhausted():
    good = build_lab("tool_agent", 1)
    broken = good.reference_solution.replace("tools=self.tool_schemas", "tools=None")
    provider = ReplayProvider([spec_json("tool_agent", 1, reference_solution=broken)] * 2 + ['{"remarks": []}'])
    result = generate_lab("Агент с инструментами", 1, mode="llm", provider=provider, max_attempts=2)
    assert not result.report.passed
    assert any("черновик" in line for line in result.log)


def test_generator_prompt_contains_level_and_example():
    provider = ReplayProvider([spec_json("rag_agent", 3), '{"remarks": []}'])
    generate_lab("Поиск по документам", 3, mode="llm", provider=provider, materials="Лекция: BM25 и эмбеддинги")
    system, user = provider.calls[0]
    assert "ScriptedLLM" in system["content"]
    assert "Продвинутый" in user["content"]
    assert "RAG: ответы по базе знаний" in user["content"], "образец — ближайшая тема каталога"
    assert "BM25" in user["content"]


def test_heuristic_review_flags_requirement_count():
    spec = build_lab("tool_agent", 1)
    spec.level = 3
    remarks = heuristic_review(spec)
    assert any("Требований с автопроверкой" in r.comment for r in remarks)


def test_simulated_students():
    spec = build_lab("memory_agent", 1)
    provider = ReplayProvider(["```python\n" + spec.reference_solution + "```", "```python\n" + spec.starter_code + "```"])
    result = simulate_students(provider, spec, n=2)
    assert result["solved"] == 1 and result["solve_rate"] == 0.5


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Вот:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json_object('ответ {"a": 3} конец') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json_object("нет json")


def test_openai_compatible_provider(monkeypatch):
    import httpx

    from labgen.llm import OpenAICompatibleProvider

    sent = []
    replies = [httpx.Response(503, text="перегружен"), httpx.Response(200, json={
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    })]

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append((url, json, headers))
        return replies.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    provider = OpenAICompatibleProvider(base_url="http://localhost:11434/v1/", model="qwen", api_key="k")
    assert provider.complete([{"role": "user", "content": "hi"}], json_mode=True) == '{"ok": true}'
    url, payload, headers = sent[-1]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert payload["response_format"] == {"type": "json_object"} and payload["model"] == "qwen"
    assert headers["Authorization"] == "Bearer k"
    assert len(sent) == 2, "ответ 503 нужно повторить"
    assert provider.usage["prompt_tokens"] == 10

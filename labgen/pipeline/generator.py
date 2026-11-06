"""Строительные блоки генерации: выбор работы из каталога, разбор ответа LLM, замечания для исправления.

Сам порядок шагов (генерация → проверка → исправление) задаётся графом LangGraph
в :mod:`labgen.agent.graph`.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..catalog import get_topic, list_topics, match_topic, score_topics
from ..models import LabSpec, ValidationReport
from .jsonutil import extract_json_object


class TopicNotFound(LookupError):
    def __init__(self, request: str) -> None:
        names = "; ".join(t.title for t in list_topics())
        super().__init__(
            f"Тема «{request}» не найдена в каталоге. Доступные темы: {names}. "
            "Подключите LLM в настройках аккаунта, чтобы генерировать работы по произвольным темам."
        )


def catalog_spec(request: str, level: int, topic_id: str | None = None) -> LabSpec:
    topic = get_topic(topic_id) if topic_id else match_topic(request)
    if topic is None:
        raise TopicNotFound(request)
    return topic.build(level)


def pick_example(request: str, level: int) -> LabSpec:
    """Образец для few-shot: ближайшая тема каталога того же уровня."""
    best, score = score_topics(request)[0]
    topic = best if score > 0 else get_topic("tool_agent")
    return topic.build(level)


def parse_spec(raw: str, level: int) -> LabSpec:
    data = extract_json_object(raw)
    data["level"] = level
    data.setdefault("topic_id", None)
    return LabSpec.model_validate(data)


def parse_error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()[:3]
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors)
    return str(exc).splitlines()[0][:300]


def report_line(report: ValidationReport) -> str:
    score = report.mutation_score
    mutation = f", мутанты {score:.0%}" if score is not None else ""
    status = "пройдена" if report.passed else f"не пройдена ({len(report.errors())} ошибок)"
    return f"Проверка {status}: покрытие {report.auto_coverage:.0%}{mutation}, {report.duration_s:.1f} с."


def validation_issues(report: ValidationReport) -> list[str]:
    issues = [f"{c.name}: {c.details}" for c in report.errors()]
    for m in report.mutants:
        if m.applied and not m.killed:
            issues.append(f"Мутант {m.id} «{m.description}» не обнаружен тестами — добавь тест.")
        if not m.applied:
            issues.append(f"Мутант {m.id}: фрагмент find не найден в эталоне.")
    return issues


def quality(report: ValidationReport) -> tuple:
    return (report.passed, -len(report.errors()), report.mutation_score or 0.0, report.reference_pass_rate)

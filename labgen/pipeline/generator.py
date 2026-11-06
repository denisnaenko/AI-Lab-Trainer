"""Генераторы лабораторных работ.

* :class:`CatalogGenerator` — офлайн: выбирает подходящую тему каталога по запросу преподавателя.
* :class:`LLMGenerator` — создаёт новую работу с помощью LLM и чинит её по замечаниям
  валидатора (цикл «генерация → проверка → исправление», как в PyTaskSyn и AutoCode).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..catalog import get_topic, list_topics, match_topic, score_topics
from ..levels import get_level
from ..llm.base import LLMError, LLMProvider
from ..models import LabSpec, ValidationReport
from . import prompts
from .jsonutil import extract_json_object
from .validator import validate_spec


class TopicNotFound(LookupError):
    def __init__(self, request: str) -> None:
        names = "; ".join(f"{t.title}" for t in list_topics())
        super().__init__(
            f"Тема «{request}» не найдена в каталоге. Доступные темы: {names}. "
            "Подключите LLM, чтобы генерировать работы по произвольным темам."
        )


@dataclass
class GeneratorOutput:
    spec: LabSpec
    report: ValidationReport
    log: list[str] = field(default_factory=list)
    attempts: int = 1


class CatalogGenerator:
    def generate(self, request: str, level: int, topic_id: str | None = None) -> GeneratorOutput:
        topic = get_topic(topic_id) if topic_id else match_topic(request)
        if topic is None:
            raise TopicNotFound(request)
        spec = topic.build(level)
        log = [f"Каталог: выбрана тема «{topic.title}» (уровень {level})."]
        report = validate_spec(spec)
        log.append(_report_line(report))
        return GeneratorOutput(spec=spec, report=report, log=log)


def parse_spec(raw: str, level: int) -> LabSpec:
    data = extract_json_object(raw)
    data["level"] = level
    data.setdefault("topic_id", None)
    return LabSpec.model_validate(data)


def _report_line(report: ValidationReport) -> str:
    score = report.mutation_score
    mutation = f", мутанты {score:.0%}" if score is not None else ""
    status = "пройдена" if report.passed else f"не пройдена ({len(report.errors())} ошибок)"
    return f"Проверка {status}: покрытие {report.auto_coverage:.0%}{mutation}, {report.duration_s:.1f} с."


def _issues(report: ValidationReport) -> list[str]:
    issues = [f"{c.name}: {c.details}" for c in report.errors()]
    for m in report.mutants:
        if m.applied and not m.killed:
            issues.append(f"Мутант {m.id} «{m.description}» не обнаружен тестами — добавь тест.")
        if not m.applied:
            issues.append(f"Мутант {m.id}: фрагмент find не найден в эталоне.")
    return issues


class LLMGenerator:
    def __init__(self, provider: LLMProvider, max_attempts: int = 3) -> None:
        self.provider = provider
        self.max_attempts = max(1, max_attempts)

    @staticmethod
    def pick_example(request: str, level: int) -> LabSpec:
        """Образец для few-shot: ближайшая тема каталога того же уровня."""
        best, score = score_topics(request)[0]
        topic = best if score > 0 else get_topic("tool_agent")
        return topic.build(level)

    def generate(self, request: str, level: int, materials: str | None = None) -> GeneratorOutput:
        lvl = get_level(level)
        example = self.pick_example(request, level)
        log = [f"LLM ({self.provider.name}): образец — «{example.title}», уровень {level}."]
        messages = prompts.generation_messages(request, lvl, example, materials)

        best: GeneratorOutput | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.provider.complete(messages, json_mode=True)
            except LLMError as exc:
                log.append(f"Попытка {attempt}: ошибка модели — {exc}")
                if best is None:
                    raise
                break
            try:
                spec = parse_spec(raw, level)
            except (ValueError, ValidationError) as exc:
                text = str(exc).splitlines()[0][:300]
                log.append(f"Попытка {attempt}: ответ не разобран — {text}")
                messages = prompts.repair_messages(
                    request, lvl, raw[:20000], [f"Ответ не является корректной спецификацией: {text}"]
                )
                continue
            report = validate_spec(spec)
            log.append(f"Попытка {attempt}: " + _report_line(report))
            output = GeneratorOutput(spec=spec, report=report, log=log, attempts=attempt)
            if best is None or _quality(report) > _quality(best.report):
                best = output
            if report.passed:
                return output
            issues = _issues(report)
            log.append("Замечания для исправления: " + "; ".join(issues)[:600])
            spec_json = json.dumps(spec.model_dump(exclude={"topic_id", "level"}), ensure_ascii=False, indent=1)
            messages = prompts.repair_messages(request, lvl, spec_json, issues)

        if best is None:
            raise LLMError("Модель не вернула ни одной корректной спецификации: " + " | ".join(log[-3:]))
        best.log = log
        log.append("Работа сохранена как черновик с ошибками проверки — исправьте её в редакторе.")
        return best


def _quality(report: ValidationReport) -> tuple:
    return (report.passed, -len(report.errors()), report.mutation_score or 0.0, report.reference_pass_rate)

"""Рецензирование лабораторной работы перед публикацией.

Рецензент не ставит баллов, а готовит замечания для преподавателя: согласие экспертов
с баллами LLM-судей заметно ниже, чем с их обоснованиями (Judging the Judges, 2026).
Без LLM работает набор формальных правил.
"""

from __future__ import annotations

from ..levels import get_level
from ..llm.base import LLMError, LLMProvider
from ..models import LabSpec, ReviewRemark, test_names, requirement_of
from . import prompts
from .jsonutil import extract_json_object


def heuristic_review(spec: LabSpec) -> list[ReviewRemark]:
    level = get_level(spec.level)
    remarks: list[ReviewRemark] = []
    auto = [r for r in spec.requirements if r.auto]
    if not level.min_auto_requirements <= len(auto) <= level.max_auto_requirements:
        remarks.append(
            ReviewRemark(
                severity="minor",
                section="требования",
                comment=(
                    f"Требований с автопроверкой {len(auto)}, для уровня «{level.name}» ожидается "
                    f"{level.min_auto_requirements}–{level.max_auto_requirements}. Проверьте, соответствует ли объём уровню."
                ),
            )
        )
    public_reqs = {requirement_of(n) for n in test_names(spec.tests_public)}
    without_public = [r.id for r in auto if r.id not in public_reqs]
    if without_public:
        remarks.append(
            ReviewRemark(
                severity="info",
                section="тесты",
                comment=(
                    "Нет публичных тестов для " + ", ".join(without_public)
                    + ": студент узнает о проверке этих требований только после сдачи."
                ),
            )
        )
    if "solution.py" not in spec.statement_md:
        remarks.append(
            ReviewRemark(severity="minor", section="условие", comment="В условии не указано, что решение пишется в solution.py.")
        )
    if spec.estimated_minutes is None:
        remarks.append(
            ReviewRemark(severity="info", section="условие", comment="Не указано ориентировочное время выполнения.")
        )
    if "TODO" not in spec.starter_code and spec.level == 1:
        remarks.append(
            ReviewRemark(
                severity="minor",
                section="шаблон",
                comment="Для базового уровня в шаблоне ожидаются пошаговые подсказки TODO.",
            )
        )
    return remarks


def llm_review(provider: LLMProvider, spec: LabSpec) -> list[ReviewRemark]:
    raw = provider.complete(prompts.review_messages(spec, get_level(spec.level)), json_mode=True)
    data = extract_json_object(raw)
    remarks = []
    for item in data.get("remarks", [])[:8]:
        try:
            remarks.append(ReviewRemark.model_validate(item))
        except Exception:
            continue
    return remarks


def review_spec(spec: LabSpec, provider: LLMProvider | None = None, log: list[str] | None = None) -> list[ReviewRemark]:
    remarks = heuristic_review(spec)
    if provider is not None:
        try:
            llm_remarks = llm_review(provider, spec)
            remarks = llm_remarks + remarks
            if log is not None:
                log.append(f"Рецензент LLM: {len(llm_remarks)} замечаний.")
        except (LLMError, ValueError) as exc:
            if log is not None:
                log.append(f"Рецензент LLM недоступен: {exc}")
    return remarks

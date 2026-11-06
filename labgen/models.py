"""Модель данных лабораторной работы.

Связь «требование ↔ тест» задаётся соглашением об именах: тест
``test_r3_handles_unknown_tool`` проверяет требование ``R3``. Это делает
спецификацию проверяемой автоматически и понятной LLM-генератору.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TEST_NAME_RE = re.compile(r"^test_(r\d+)_", re.IGNORECASE)


class Requirement(BaseModel):
    id: str = Field(description="Идентификатор вида R1, R2, ...")
    text: str
    points: int = Field(ge=0)
    auto: bool = Field(default=True, description="Проверяется автотестами")

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        value = value.strip().upper()
        if not re.fullmatch(r"R\d+", value):
            raise ValueError(f"Идентификатор требования должен иметь вид R1, R2...: {value!r}")
        return value


class Mutant(BaseModel):
    """Намеренно сломанная версия эталона: тесты обязаны её «убить»."""

    id: str
    description: str
    find: str
    replace: str


class LabSpec(BaseModel):
    topic_id: str | None = None
    title: str
    level: int = Field(ge=1, le=3)
    summary: str
    statement_md: str
    starter_code: str
    reference_solution: str
    tests_public: str
    tests_hidden: str
    requirements: list[Requirement]
    mutants: list[Mutant] = Field(default_factory=list)
    estimated_minutes: int | None = None

    # --- производные свойства -------------------------------------------------

    @property
    def total_points(self) -> int:
        return sum(r.points for r in self.requirements)

    def requirement(self, req_id: str) -> Requirement | None:
        req_id = req_id.upper()
        return next((r for r in self.requirements if r.id == req_id), None)

    def tests_by_requirement(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {r.id: [] for r in self.requirements}
        for name in test_names(self.tests_public) + test_names(self.tests_hidden):
            req = requirement_of(name)
            if req is not None:
                mapping.setdefault(req, []).append(name)
        return mapping

    def auto_coverage(self) -> float:
        """Доля баллов, которые проверяются автотестами (критерий проекта: ≥ 80%)."""
        total = self.total_points
        if total == 0:
            return 0.0
        mapping = self.tests_by_requirement()
        covered = sum(r.points for r in self.requirements if r.auto and mapping.get(r.id))
        return covered / total


class ValidationCheck(BaseModel):
    name: str
    ok: bool
    details: str = ""
    severity: Literal["error", "warning"] = "error"


class MutantResult(BaseModel):
    id: str
    description: str
    killed: bool
    applied: bool = True
    killed_by: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    checks: list[ValidationCheck] = Field(default_factory=list)
    mutants: list[MutantResult] = Field(default_factory=list)
    auto_coverage: float = 0.0
    reference_pass_rate: float = 0.0
    starter_fail_rate: float = 0.0
    duration_s: float = 0.0
    created_at: str = Field(default_factory=lambda: now_iso())

    @property
    def mutation_score(self) -> float | None:
        applied = [m for m in self.mutants if m.applied]
        if not applied:
            return None
        return sum(m.killed for m in applied) / len(applied)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.severity == "error")

    def errors(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    def warnings(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.ok and c.severity == "warning"]


class ReviewRemark(BaseModel):
    severity: Literal["info", "minor", "major"] = "minor"
    section: str = "общее"
    comment: str


class RequirementGrade(BaseModel):
    id: str
    text: str
    points: int
    earned: float
    auto: bool
    tests_total: int = 0
    tests_passed: int = 0
    failures: list[str] = Field(default_factory=list)


class GradeReport(BaseModel):
    lab_title: str
    level: int
    student: str | None = None
    requirements: list[RequirementGrade]
    auto_score: float
    auto_max: int
    manual_max: int
    total_max: int
    tests_passed: int
    tests_total: int
    timed_out: bool = False
    infrastructure_error: str | None = None
    feedback: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: now_iso())

    @property
    def auto_percent(self) -> float:
        return 100.0 * self.auto_score / self.auto_max if self.auto_max else 0.0


class LabRecord(BaseModel):
    """Лабораторная работа в хранилище: спецификация + статус + отчёты."""

    id: str
    request_topic: str
    level: int
    mode: str
    status: Literal["draft", "approved"] = "draft"
    version: int = 1
    spec: LabSpec
    validation: ValidationReport | None = None
    review: list[ReviewRemark] = Field(default_factory=list)
    generation_log: list[str] = Field(default_factory=list)
    simulation: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: now_iso())
    updated_at: str = Field(default_factory=lambda: now_iso())


# --- утилиты ------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def test_names(code: str) -> list[str]:
    """Имена тестовых функций верхнего уровня в модуле (без запуска кода)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def requirement_of(test_name: str) -> str | None:
    match = TEST_NAME_RE.match(test_name)
    return match.group(1).upper() if match else None


# pytest не должен собирать эти утилиты как тесты, если модуль импортируют в тестах
test_names.__test__ = False  # type: ignore[attr-defined]

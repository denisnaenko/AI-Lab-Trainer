"""Core data model: a lab assignment and everything generated for it.

A `LabAssignment` is the single source of truth. Generation fills it in, the teacher
edits it (as files on disk, see `storage.py`), validation checks it, exporters publish it,
and the grader runs its tests against student submissions.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Difficulty(StrEnum):
    """Three levels required by the acceptance criteria.

    The level changes how much scaffolding the starter gives, how many requirements
    there are and how much of the test suite is hidden from students.
    """

    BASIC = "basic"  # most code given, student fills in marked TODOs; all tests visible
    INTERMEDIATE = "intermediate"  # skeleton + interfaces; some hidden tests
    ADVANCED = "advanced"  # only the task and the public API contract; most tests hidden


class Status(StrEnum):
    DRAFT = "draft"  # produced by the generator, not reviewed yet
    APPROVED = "approved"  # teacher reviewed it; only approved labs can be exported
    PUBLISHED = "published"


class CheckKind(StrEnum):
    AUTO = "auto"  # covered by at least one automated test
    MANUAL = "manual"  # needs a human (e.g. quality of a written report)


class Requirement(BaseModel):
    """One checkable thing the student must do. Rubric and tests both point at these."""

    id: str = Field(pattern=r"^R\d+$")
    text: str
    check: CheckKind = CheckKind.AUTO
    test_ids: list[str] = Field(default_factory=list)


class AutoTest(BaseModel):
    """A pytest test that verifies one or more requirements.

    Tests for agentic code never call a real LLM: they use the scripted `FakeLLM`
    fixture shipped in the starter, so grading is deterministic and needs no API key.
    """

    id: str
    requirement_ids: list[str]
    file: str  # path of one of `LabAssignment.test_files`
    hidden: bool = False  # hidden tests are not shipped to students
    points: float = 1.0


class RubricItem(BaseModel):
    requirement_id: str
    points: float
    criteria: str  # what full / partial / zero credit looks like


class LabFile(BaseModel):
    """A file in the starter kit or in the reference solution."""

    path: str
    content: str


class LabAssignment(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    topic: str  # the teacher's original prompt, e.g. "Создание агента с инструментами"
    difficulty: Difficulty
    language: str = "ru"
    status: Status = Status.DRAFT
    estimated_minutes: int = 90

    brief: str  # markdown: goal and context of the lab
    instructions: str  # markdown: step-by-step instructions for the student
    learning_objectives: list[str]
    requirements: list[Requirement]
    starter_files: list[LabFile]
    solution_files: list[LabFile]  # reference solution; never shipped to students
    test_files: list[LabFile]  # pytest files (+ conftest with FakeLLM); `AutoTest.file` points here
    tests: list[AutoTest]
    rubric: list[RubricItem]

    @model_validator(mode="after")
    def _check_references(self) -> LabAssignment:
        req_ids = {r.id for r in self.requirements}
        test_ids = {t.id for t in self.tests}
        for r in self.requirements:
            if missing := set(r.test_ids) - test_ids:
                raise ValueError(f"{r.id} references unknown tests {sorted(missing)}")
        for t in self.tests:
            if missing := set(t.requirement_ids) - req_ids:
                raise ValueError(f"test {t.id} references unknown requirements {sorted(missing)}")
        test_paths = {f.path for f in self.test_files}
        for t in self.tests:
            if t.file not in test_paths:
                raise ValueError(f"test {t.id} points at missing file {t.file}")
        for item in self.rubric:
            if item.requirement_id not in req_ids:
                raise ValueError(f"rubric references unknown requirement {item.requirement_id}")
        return self

    @property
    def auto_check_coverage(self) -> float:
        """Share of requirements that are automatically checked. Target: >= 0.8."""
        if not self.requirements:
            return 0.0
        auto = [r for r in self.requirements if r.check is CheckKind.AUTO and r.test_ids]
        return len(auto) / len(self.requirements)

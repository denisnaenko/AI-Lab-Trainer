"""Prove a generated (or teacher-edited) lab is internally consistent.

Used by the `validate` node of the generation graph and by `lab-trainer validate/approve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lab_trainer.grading.runner import run_tests as grade
from lab_trainer.models import LabAssignment

MIN_AUTO_COVERAGE = 0.8
_OUTPUT_LIMIT = 6000  # tail of pytest output fed back to the repair node


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    output: str = ""  # pytest output of the failing runs

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_lab(lab: LabAssignment, run_tests: bool = True) -> ValidationReport:
    report = ValidationReport()
    if lab.auto_check_coverage < MIN_AUTO_COVERAGE:
        report.errors.append(
            f"auto-check coverage {lab.auto_check_coverage:.0%} is below {MIN_AUTO_COVERAGE:.0%}"
        )
    if not run_tests:
        return report

    solution = grade(lab, lab.solution_files)
    if failed := [r.requirement_id for r in solution.requirements if not r.passed]:
        report.errors.append(f"reference solution fails tests for {', '.join(failed)}")
        report.output += solution.raw_output
    # otherwise the task is trivially solved by the starter
    starter = grade(lab, lab.starter_files)
    if passed := [r.requirement_id for r in starter.requirements if r.points > 0]:
        report.errors.append(f"starter already passes tests for {', '.join(passed)}")
        report.output += starter.raw_output
    report.output = report.output[-_OUTPUT_LIMIT:]
    return report

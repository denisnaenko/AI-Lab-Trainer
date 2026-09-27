"""Stage 4: prove a generated (or teacher-edited) lab is internally consistent."""

from __future__ import annotations

from dataclasses import dataclass, field

from lab_trainer.models import LabAssignment

MIN_AUTO_COVERAGE = 0.8


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_lab(lab: LabAssignment, run_tests: bool = True) -> ValidationReport:
    report = ValidationReport()
    if lab.auto_check_coverage < MIN_AUTO_COVERAGE:
        report.errors.append(
            f"auto-check coverage {lab.auto_check_coverage:.0%} is below {MIN_AUTO_COVERAGE:.0%}"
        )
    if run_tests:
        # TODO: grading.runner.run_tests(lab, lab.solution_files) must pass every test;
        # run_tests(lab, lab.starter_files) must fail every auto-checked requirement,
        # otherwise the task is trivially solved by the starter.
        pass
    return report

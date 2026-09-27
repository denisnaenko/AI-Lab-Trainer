"""Run a lab's tests against a set of files and score the result per requirement.

Used for three things: validating the reference solution, checking that the starter
does not already pass, and grading student submissions (from Moodle or Gitverse CI).

`AutoTest.id` is the pytest function name; results are read from pytest's JUnit XML.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel

from lab_trainer.models import CheckKind, LabAssignment, LabFile


class RequirementResult(BaseModel):
    requirement_id: str
    passed: bool
    points: float
    max_points: float


class GradeReport(BaseModel):
    lab_id: str
    requirements: list[RequirementResult]
    manual_requirement_ids: list[str]  # left for the teacher to grade
    raw_output: str

    @property
    def score(self) -> float:
        return sum(r.points for r in self.requirements)

    @property
    def max_score(self) -> float:
        return sum(r.max_points for r in self.requirements)


def run_tests(
    lab: LabAssignment, files: list[LabFile], include_hidden: bool = True, timeout: int = 120
) -> GradeReport:
    """Copy `files` to a temp dir and the lab tests to its tests/, then run pytest.

    TODO: run inside a container (no network, CPU/memory limits) instead of a bare
    subprocess before grading untrusted student code.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        placed = [(f, work / f.path) for f in files]
        placed += [(f, work / "tests" / f.path) for f in lab.test_files]
        for f, dest in placed:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content, encoding="utf-8")
        tests = [t for t in lab.tests if include_hidden or not t.hidden]
        junit = work / "report.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}",
             *sorted({f"tests/{t.file}" for t in tests})],
            cwd=work, capture_output=True, text=True, timeout=timeout, check=False,
        )
        passed = _passed_tests(junit) if junit.exists() else set()

    results = []
    for req in lab.requirements:
        if req.check is not CheckKind.AUTO:
            continue
        req_tests = [t for t in tests if req.id in t.requirement_ids]
        if not req_tests:
            continue
        max_points = sum(t.points for t in req_tests)
        points = sum(t.points for t in req_tests if t.id in passed)
        results.append(RequirementResult(
            requirement_id=req.id, passed=points == max_points,
            points=points, max_points=max_points,
        ))
    return GradeReport(
        lab_id=lab.id,
        requirements=results,
        manual_requirement_ids=[r.id for r in lab.requirements if r.check is CheckKind.MANUAL],
        raw_output=proc.stdout + proc.stderr,
    )


def _passed_tests(junit: Path) -> set[str]:
    """Names of tests whose every case passed (a parametrized test counts as one)."""
    seen, failed = set(), set()
    for case in ET.parse(junit).iter("testcase"):
        name = case.get("name", "").split("[")[0]
        seen.add(name)
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            failed.add(name)
    return seen - failed

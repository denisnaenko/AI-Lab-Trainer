"""End-to-end checks on the hand-written example lab: format, grading, export gate."""

from pathlib import Path

import pytest

from lab_trainer.export import GitverseRepoExporter, NotApprovedError
from lab_trainer.generation.validation import validate_lab
from lab_trainer.grading import run_tests
from lab_trainer.models import Status
from lab_trainer.storage import load_lab, save_lab

EXAMPLE = Path(__file__).parents[1] / "labs" / "tool-agent-basic"


@pytest.fixture
def lab():
    return load_lab(EXAMPLE)


def test_example_lab_is_valid(lab):
    assert validate_lab(lab, run_tests=False).ok
    assert lab.auto_check_coverage >= 0.8


def test_roundtrip_through_disk(lab, tmp_path):
    assert load_lab(save_lab(lab, tmp_path)) == lab


def test_reference_solution_passes_everything(lab):
    report = run_tests(lab, lab.solution_files)
    assert report.score == report.max_score, report.raw_output
    assert report.manual_requirement_ids == ["R6"]


def test_starter_passes_nothing(lab):
    report = run_tests(lab, lab.starter_files)
    assert report.score == 0, report.raw_output


def test_draft_cannot_be_exported(lab, tmp_path):
    with pytest.raises(NotApprovedError):
        GitverseRepoExporter().export(lab, tmp_path)


def test_gitverse_export_hides_solution(lab, tmp_path):
    lab = lab.model_copy(update={"status": Status.APPROVED})
    repo = Path(GitverseRepoExporter().export(lab, tmp_path).location)
    shipped = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()}
    assert shipped == {"README.md", "agent.py", "tests/conftest.py", "tests/test_agent.py"}
    assert "ast.parse" not in (repo / "agent.py").read_text(encoding="utf-8")

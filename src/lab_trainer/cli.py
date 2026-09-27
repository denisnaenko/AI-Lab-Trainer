"""Teacher-facing CLI: generate -> (edit files) -> validate -> approve -> export -> grade."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from lab_trainer.export import EXPORTERS
from lab_trainer.generation.validation import validate_lab
from lab_trainer.grading import run_tests
from lab_trainer.models import Difficulty, LabFile, Status
from lab_trainer.storage import load_lab, save_lab

app = typer.Typer(help="Generate, review, publish and auto-grade lab assignments.")
LABS = Path("labs")


@app.command()
def generate(topic: str, difficulty: Difficulty = Difficulty.BASIC, notes: str = "") -> None:
    """Generate a draft lab from a topic and save it under labs/ for review."""
    from lab_trainer.generation import GenerationPipeline, GenerationRequest
    from lab_trainer.generation.llm import ClaudeClient

    lab = GenerationPipeline(ClaudeClient()).run(
        GenerationRequest(topic=topic, difficulty=difficulty, teacher_notes=notes)
    )
    typer.echo(f"Draft saved to {save_lab(lab, LABS)}. Review and edit it, then run approve.")


@app.command()
def validate(lab_id: str) -> None:
    """Re-check a lab after the teacher edited it."""
    lab = load_lab(LABS / lab_id)
    report = validate_lab(lab)
    typer.echo(f"auto-check coverage: {lab.auto_check_coverage:.0%}")
    for err in report.errors:
        typer.echo(f"error: {err}")
    raise typer.Exit(0 if report.ok else 1)


@app.command()
def approve(lab_id: str) -> None:
    """Mark a reviewed lab as approved so it can be exported."""
    lab = load_lab(LABS / lab_id)
    report = validate_lab(lab)
    if not report.ok:
        typer.echo("\n".join(report.errors))
        raise typer.Exit(1)
    lab_yaml = LABS / lab_id / "lab.yaml"
    meta = yaml.safe_load(lab_yaml.read_text(encoding="utf-8"))
    meta["status"] = Status.APPROVED.value
    lab_yaml.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    typer.echo(f"{lab_id} approved")


@app.command()
def export(lab_id: str, target: str = "gitverse", out: Path = Path("out")) -> None:
    """Export an approved lab to Moodle (XML import) or Gitverse (template repo)."""
    result = EXPORTERS[target]().export(load_lab(LABS / lab_id), out)
    typer.echo(f"{result.target}: {result.location}")
    for note in result.notes:
        typer.echo(f"note: {note}")


@app.command()
def grade(lab_id: str, submission: Path) -> None:
    """Grade a student's submission directory against the lab's tests."""
    lab = load_lab(LABS / lab_id)
    files = [
        LabFile(path=p.relative_to(submission).as_posix(), content=p.read_text(encoding="utf-8"))
        for p in submission.rglob("*.py")
        if "tests" not in p.relative_to(submission).parts
    ]
    report = run_tests(lab, files)
    for r in report.requirements:
        typer.echo(f"{r.requirement_id}: {r.points:g}/{r.max_points:g}")
    typer.echo(f"total: {report.score:g}/{report.max_score:g}")
    if report.manual_requirement_ids:
        typer.echo(f"manual review: {', '.join(report.manual_requirement_ids)}")


if __name__ == "__main__":
    app()

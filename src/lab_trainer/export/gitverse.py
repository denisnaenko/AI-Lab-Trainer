"""Gitverse export: the lab becomes a template repository with CI auto-grading.

Layout produced in out_dir/<lab-id>/:
    README.md                  brief + instructions
    <starter files>
    tests/                     visible tests only
    .gitverse/workflows/grade.yaml   CI job: installs deps, runs `lab-trainer grade`,
                                     which also pulls hidden tests from a teacher-only repo

Publishing is a plain `git push` to a Gitverse remote, so the local export already
counts as a "correct import" even before the API client below is finished.
"""

from __future__ import annotations

from pathlib import Path

from lab_trainer.export.base import Exporter, ExportResult, student_files
from lab_trainer.models import LabAssignment


class GitverseRepoExporter(Exporter):
    target = "gitverse"

    def _export(self, lab: LabAssignment, out_dir: Path) -> ExportResult:
        repo = out_dir / lab.id
        for f in student_files(lab):
            dest = repo / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content, encoding="utf-8")
        (repo / "README.md").write_text(
            f"# {lab.title}\n\n{lab.brief}\n\n{lab.instructions}\n", encoding="utf-8"
        )
        # TODO: CI workflow file; confirm Gitverse CI workflow syntax and path.
        return ExportResult(
            target=self.target, location=str(repo), notes=["CI workflow not generated yet"]
        )


class GitverseClient:
    """TODO: create repo from template, add students, read CI results via the Gitverse API."""

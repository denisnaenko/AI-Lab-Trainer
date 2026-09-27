"""Common exporter contract. Every target refuses labs the teacher has not approved."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from lab_trainer.models import LabAssignment, LabFile, Status


class ExportResult(BaseModel):
    target: str
    location: str  # file path or URL of what was produced/published
    notes: list[str] = []


class NotApprovedError(Exception):
    pass


class Exporter(ABC):
    target: str

    def export(self, lab: LabAssignment, out_dir: Path) -> ExportResult:
        if lab.status is Status.DRAFT:
            raise NotApprovedError(
                f"{lab.id} is still a draft; review it and run `lab-trainer approve {lab.id}`"
            )
        return self._export(lab, out_dir)

    @abstractmethod
    def _export(self, lab: LabAssignment, out_dir: Path) -> ExportResult: ...


def student_files(lab: LabAssignment) -> list[LabFile]:
    """What students may see: starter + visible tests. Never the solution or hidden tests."""
    hidden = {t.file for t in lab.tests if t.hidden}
    visible_tests = [
        LabFile(path=f"tests/{f.path}", content=f.content)
        for f in lab.test_files
        if f.path not in hidden
    ]
    return [*lab.starter_files, *visible_tests]

"""Файловое хранилище лабораторных работ с версиями и результатами проверки.

Структура::

    data/labs/<lab_id>/lab.json            — текущая версия (LabRecord)
    data/labs/<lab_id>/versions/v0001.json — снимки всех версий
    data/labs/<lab_id>/grades/*.json       — результаты автопроверки решений

Каждое сохранение создаёт новую версию: можно восстановить, кто и что менял,
и воспроизвести генерацию (в записи хранится журнал конвейера).
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

from .exporters import slugify
from .models import GradeReport, LabRecord, now_iso

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,80}$")


class LabNotFound(KeyError):
    pass


class LabStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.labs_dir = self.root / "labs"
        self.labs_dir.mkdir(parents=True, exist_ok=True)

    # --- пути -------------------------------------------------------------------

    def _dir(self, lab_id: str) -> Path:
        if not ID_RE.match(lab_id):
            raise LabNotFound(lab_id)
        return self.labs_dir / lab_id

    def new_id(self, title: str, level: int) -> str:
        return f"{slugify(title, 30)}-l{level}-{uuid.uuid4().hex[:6]}"

    # --- запись -----------------------------------------------------------------

    def save(self, record: LabRecord, *, new_version: bool = True) -> LabRecord:
        folder = self._dir(record.id)
        (folder / "versions").mkdir(parents=True, exist_ok=True)
        if new_version and (folder / "lab.json").exists():
            record.version = self.get(record.id).version + 1
        record.updated_at = now_iso()
        payload = record.model_dump_json(indent=2)
        (folder / "lab.json").write_text(payload, encoding="utf-8")
        (folder / "versions" / f"v{record.version:04d}.json").write_text(payload, encoding="utf-8")
        return record

    def delete(self, lab_id: str) -> None:
        folder = self._dir(lab_id)
        if folder.exists():
            shutil.rmtree(folder)

    # --- чтение -----------------------------------------------------------------

    def get(self, lab_id: str) -> LabRecord:
        path = self._dir(lab_id) / "lab.json"
        if not path.is_file():
            raise LabNotFound(lab_id)
        return LabRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, owner_id: int | None = None, *, include_shared: bool = True) -> list[LabRecord]:
        """Работы пользователя (и общие, без владельца). ``owner_id=None`` — все работы."""
        records = []
        for path in self.labs_dir.glob("*/lab.json"):
            try:
                records.append(LabRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        if owner_id is not None:
            records = [r for r in records if r.owner_id == owner_id or (include_shared and r.owner_id is None)]
        return sorted(records, key=lambda r: r.updated_at, reverse=True)

    def versions(self, lab_id: str) -> list[int]:
        folder = self._dir(lab_id) / "versions"
        return sorted(int(p.stem[1:]) for p in folder.glob("v*.json"))

    def get_version(self, lab_id: str, version: int) -> LabRecord:
        path = self._dir(lab_id) / "versions" / f"v{version:04d}.json"
        if not path.is_file():
            raise LabNotFound(f"{lab_id} v{version}")
        return LabRecord.model_validate_json(path.read_text(encoding="utf-8"))

    # --- результаты проверки ------------------------------------------------------

    def save_grade(self, lab_id: str, report: GradeReport) -> str:
        folder = self._dir(lab_id) / "grades"
        folder.mkdir(parents=True, exist_ok=True)
        student = slugify(report.student or "student", 30)
        grade_id = f"{now_iso().replace(':', '').replace('+', '_')}-{student}-{uuid.uuid4().hex[:4]}"
        (folder / f"{grade_id}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return grade_id

    def list_grades(self, lab_id: str) -> list[tuple[str, GradeReport]]:
        folder = self._dir(lab_id) / "grades"
        result = []
        for path in sorted(folder.glob("*.json"), reverse=True):
            result.append((path.stem, GradeReport.model_validate_json(path.read_text(encoding="utf-8"))))
        return result

    def get_grade(self, lab_id: str, grade_id: str) -> GradeReport:
        if not re.match(r"^[\w.-]+$", grade_id):
            raise LabNotFound(grade_id)
        path = self._dir(lab_id) / "grades" / f"{grade_id}.json"
        if not path.is_file():
            raise LabNotFound(grade_id)
        return GradeReport.model_validate_json(path.read_text(encoding="utf-8"))


def export_json(record: LabRecord) -> str:
    return json.dumps(json.loads(record.model_dump_json()), ensure_ascii=False, indent=2)

"""On-disk format of a lab. This folder is what the teacher reviews and edits.

    labs/<lab-id>/
        lab.yaml            metadata, requirements, tests index, rubric
        brief.md            goal and context
        instructions.md     step-by-step student instructions
        starter/...         files students receive
        solution/...        reference solution (teacher only)
        tests/...           pytest files, including hidden ones

Plain Markdown + YAML + code keeps the review step tool-free: any editor or a git
diff works, and `lab-trainer validate` re-checks the lab after edits.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lab_trainer.models import LabAssignment, LabFile

_TEXT_FIELDS = {"brief": "brief.md", "instructions": "instructions.md"}
_FILE_DIRS = {"starter_files": "starter", "solution_files": "solution", "test_files": "tests"}


def save_lab(lab: LabAssignment, root: Path) -> Path:
    lab_dir = root / lab.id
    lab_dir.mkdir(parents=True, exist_ok=True)

    meta = lab.model_dump(mode="json", exclude=set(_TEXT_FIELDS) | set(_FILE_DIRS))
    (lab_dir / "lab.yaml").write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    for field, name in _TEXT_FIELDS.items():
        (lab_dir / name).write_text(getattr(lab, field), encoding="utf-8")
    for field, dirname in _FILE_DIRS.items():
        for f in getattr(lab, field):
            path = lab_dir / dirname / f.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f.content, encoding="utf-8")
    return lab_dir


def load_lab(lab_dir: Path) -> LabAssignment:
    data = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
    for field, name in _TEXT_FIELDS.items():
        data[field] = (lab_dir / name).read_text(encoding="utf-8")
    for field, dirname in _FILE_DIRS.items():
        base = lab_dir / dirname
        data[field] = [
            LabFile(path=p.relative_to(base).as_posix(), content=p.read_text(encoding="utf-8"))
            for p in sorted(base.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        ] if base.exists() else []
    return LabAssignment.model_validate(data)

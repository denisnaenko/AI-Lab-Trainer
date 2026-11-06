"""Комплект преподавателя: эталон, скрытые тесты, рубрика, отчёт о проверке задания."""

from __future__ import annotations

import json

from ..models import LabRecord, LabSpec, ValidationReport
from ..sandbox import AGENTKIT_SOURCE
from . import lab_slug, make_zip


def rubric_markdown(spec: LabSpec) -> str:
    mapping = spec.tests_by_requirement()
    lines = [
        f"# Критерии оценивания: {spec.title} (уровень {spec.level})",
        "",
        "| Требование | Баллы | Проверка | Тесты |",
        "| --- | --- | --- | --- |",
    ]
    for r in spec.requirements:
        tests = ", ".join(f"`{t}`" for t in mapping.get(r.id, [])) or "—"
        lines.append(f"| **{r.id}.** {r.text} | {r.points} | {'авто' if r.auto else 'вручную'} | {tests} |")
    lines += ["", f"Покрытие автопроверкой: {spec.auto_coverage():.0%} баллов."]
    return "\n".join(lines) + "\n"


def validation_markdown(report: ValidationReport | None) -> str:
    if report is None:
        return "# Отчёт о проверке задания\n\nПроверка не выполнялась.\n"
    lines = [
        "# Отчёт о проверке задания",
        "",
        f"Итог: **{'пройдена' if report.passed else 'есть ошибки'}**. "
        f"Покрытие автопроверкой: {report.auto_coverage:.0%}. "
        + (f"Мутационный анализ: {report.mutation_score:.0%}." if report.mutation_score is not None else ""),
        "",
        "| Проверка | Результат | Подробности |",
        "| --- | --- | --- |",
    ]
    for c in report.checks:
        status = "✅" if c.ok else ("⚠️" if c.severity == "warning" else "❌")
        lines.append(f"| {c.name} | {status} | {c.details.replace('|', '/')} |")
    if report.mutants:
        lines += ["", "## Мутанты (типичные ошибки студентов)", "", "| Мутант | Описание | Обнаружен |", "| --- | --- | --- |"]
        for m in report.mutants:
            lines.append(f"| {m.id} | {m.description} | {'да' if m.killed else 'нет'} |")
    return "\n".join(lines) + "\n"


TEACHER_README = """# Комплект преподавателя: {title} (уровень {level})

Состав:

- `lab.json` — полная спецификация работы (для `labgen grade` и повторного импорта);
- `reference/solution.py` — эталонное решение;
- `tests/test_public.py` — публичные тесты (есть у студентов);
- `tests/test_hidden.py` — скрытые тесты для итоговой проверки;
- `RUBRIC.md` — требования, баллы и связанные тесты;
- `VALIDATION.md` — отчёт автоматической проверки задания перед публикацией.

## Проверка решений

```bash
# одно решение
labgen grade lab.json path/to/solution.py
# папка с решениями (файлы *.py или подпапки с solution.py) → таблица CSV
labgen grade lab.json submissions/ --csv grades.csv
```

Без установки labgen можно проверить решение вручную: положите `solution.py` студента рядом
с `agentkit.py` и папкой `tests/` и запустите `pytest`.
"""


def teacher_kit_files(spec: LabSpec, record: LabRecord | None = None) -> dict[str, str]:
    lab_json = (record.model_dump_json(indent=2) if record else json.dumps(spec.model_dump(), ensure_ascii=False, indent=2))
    return {
        "README.md": TEACHER_README.format(title=spec.title, level=spec.level),
        "lab.json": lab_json,
        "reference/solution.py": spec.reference_solution,
        "agentkit.py": AGENTKIT_SOURCE,
        "tests/test_public.py": spec.tests_public,
        "tests/test_hidden.py": spec.tests_hidden,
        "pytest.ini": "[pytest]\npythonpath = .\ntestpaths = tests\n",
        "RUBRIC.md": rubric_markdown(spec),
        "VALIDATION.md": validation_markdown(record.validation if record else None),
    }


def export_teacher_kit(spec: LabSpec, record: LabRecord | None = None) -> tuple[str, bytes]:
    slug = lab_slug(spec)
    return f"{slug}-teacher.zip", make_zip(teacher_kit_files(spec, record), root=f"{slug}-teacher")

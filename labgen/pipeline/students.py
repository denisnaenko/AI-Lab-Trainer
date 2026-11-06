"""Симулированные студенты: LLM решает задание по условию, решения прогоняются тестами.

Доля решивших — грубая оценка понятности условия и сложности (подход PyTaskSyn, 2025).
Это ориентир для преподавателя, а не калиброванная метрика: реальную сложность
показывает только статистика по настоящим студентам.
"""

from __future__ import annotations

from ..llm.factory import LabLLM, LLMError
from ..models import LabSpec
from ..sandbox import run_tests
from . import prompts
from .jsonutil import extract_python_code


def simulate_students(llm: LabLLM, spec: LabSpec, n: int = 3) -> dict:
    files = {"test_public.py": spec.tests_public, "test_hidden.py": spec.tests_hidden}
    results = []
    for index in range(n):
        try:
            raw = llm.complete(prompts.student_messages(spec))
        except LLMError as exc:
            results.append({"student": index + 1, "error": str(exc), "pass_rate": 0.0, "solved": False})
            continue
        run = run_tests(extract_python_code(raw), files, per_test_timeout=5)
        results.append(
            {
                "student": index + 1,
                "pass_rate": round(run.pass_rate, 3),
                "solved": run.all_passed,
                "failed": run.failed_names[:10],
            }
        )
    solved = sum(r["solved"] for r in results)
    return {
        "n": n,
        "solved": solved,
        "solve_rate": solved / n if n else 0.0,
        "mean_pass_rate": sum(r["pass_rate"] for r in results) / n if n else 0.0,
        "students": results,
    }

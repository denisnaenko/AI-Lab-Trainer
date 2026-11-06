"""Автоматическая проверка лабораторной работы перед публикацией.

Проверки основаны на практиках из литературы:

* эталон проходит все тесты, а шаблон — нет (PyTaskSyn, 2025);
* тесты детерминированы (повторный прогон даёт тот же результат);
* мутационный анализ: намеренно сломанные версии эталона должны падать на тестах
  (AutoCode, 2025) — так проверяется, что тесты ловят типичные ошибки студентов;
* доля баллов под автопроверкой не меньше 80% (критерий проекта).
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..models import (
    LabSpec,
    MutantResult,
    ValidationCheck,
    ValidationReport,
    requirement_of,
    test_names,
)
from ..sandbox import RunResult, run_tests

MIN_AUTO_COVERAGE = 0.8
MIN_MUTATION_SCORE = 0.8


def _test_files(spec: LabSpec) -> dict[str, str]:
    return {"test_public.py": spec.tests_public, "test_hidden.py": spec.tests_hidden}


def _syntax_errors(spec: LabSpec) -> list[str]:
    errors = []
    for label, code in (
        ("шаблон решения", spec.starter_code),
        ("эталонное решение", spec.reference_solution),
        ("публичные тесты", spec.tests_public),
        ("скрытые тесты", spec.tests_hidden),
    ):
        try:
            compile(code, label, "exec")
        except SyntaxError as exc:
            errors.append(f"{label}: строка {exc.lineno}: {exc.msg}")
    return errors


def _structure_checks(spec: LabSpec) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    ids = [r.id for r in spec.requirements]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    problems = []
    if dup_ids:
        problems.append("повторяются идентификаторы требований: " + ", ".join(dup_ids))
    if not any(r.auto for r in spec.requirements):
        problems.append("нет ни одного требования с автопроверкой")
    if spec.total_points != 100:
        problems.append(f"сумма баллов {spec.total_points}, ожидается 100")
    for field_name in ("title", "statement_md", "starter_code", "reference_solution", "tests_hidden"):
        if not getattr(spec, field_name).strip():
            problems.append(f"пустое поле {field_name}")
    checks.append(ValidationCheck(name="Структура задания", ok=not problems, details="; ".join(problems) or "ок"))

    public = test_names(spec.tests_public)
    hidden = test_names(spec.tests_hidden)
    all_names = public + hidden
    issues = []
    duplicates = [n for n, c in Counter(all_names).items() if c > 1]
    if duplicates:
        issues.append("тесты с одинаковыми именами: " + ", ".join(duplicates))
    known = set(ids)
    for name in all_names:
        req = requirement_of(name)
        if req is None:
            issues.append(f"{name}: имя должно начинаться с test_r<N>_")
        elif req not in known:
            issues.append(f"{name}: нет требования {req}")
    mapping = spec.tests_by_requirement()
    uncovered = [r.id for r in spec.requirements if r.auto and not mapping.get(r.id)]
    if uncovered:
        issues.append("требования с автопроверкой без тестов: " + ", ".join(uncovered))
    checks.append(
        ValidationCheck(
            name="Тесты связаны с требованиями",
            ok=not issues,
            details="; ".join(issues) or f"{len(all_names)} тестов, у каждого требования есть тесты",
        )
    )
    checks.append(
        ValidationCheck(
            name="Есть публичные тесты для самопроверки",
            ok=bool(public),
            details=f"{len(public)} публичных, {len(hidden)} скрытых",
            severity="warning",
        )
    )
    coverage = spec.auto_coverage()
    checks.append(
        ValidationCheck(
            name="Покрытие автопроверкой ≥ 80%",
            ok=coverage >= MIN_AUTO_COVERAGE,
            details=f"{coverage:.0%} баллов проверяются автотестами",
        )
    )
    missing_in_statement = [r.id for r in spec.requirements if r.id not in spec.statement_md]
    checks.append(
        ValidationCheck(
            name="Все требования описаны в условии",
            ok=not missing_in_statement,
            details=("не упомянуты: " + ", ".join(missing_in_statement)) if missing_in_statement else "ок",
            severity="warning",
        )
    )
    return checks


def _failed_summary(run: RunResult, limit: int = 4) -> str:
    failed = run.failed_names
    if run.collection_error and len(failed) == len(run.expected):
        return "тесты не запустились: " + run.collection_error.splitlines()[-1][:300]
    parts = []
    for name in failed[:limit]:
        message = run.outcome(name).message.splitlines()
        parts.append(f"{name}: {message[0][:200] if message else run.outcome(name).status}")
    if len(failed) > limit:
        parts.append(f"… и ещё {len(failed) - limit}")
    return "; ".join(parts)


def _run_mutant(spec: LabSpec, mutant, files: dict[str, str], timeout: float) -> MutantResult:
    if mutant.find not in spec.reference_solution:
        return MutantResult(id=mutant.id, description=mutant.description, killed=False, applied=False)
    mutated = spec.reference_solution.replace(mutant.find, mutant.replace, 1)
    run = run_tests(mutated, files, timeout=timeout, per_test_timeout=5)
    return MutantResult(
        id=mutant.id,
        description=mutant.description,
        killed=not run.all_passed,
        killed_by=run.failed_names[:5],
    )


def validate_spec(
    spec: LabSpec,
    *,
    run_mutants: bool = True,
    check_determinism: bool = True,
    timeout: float = 60.0,
    workers: int = 4,
) -> ValidationReport:
    started = time.monotonic()
    report = ValidationReport()
    report.checks.extend(_structure_checks(spec))
    report.auto_coverage = spec.auto_coverage()

    syntax = _syntax_errors(spec)
    report.checks.append(
        ValidationCheck(name="Код без синтаксических ошибок", ok=not syntax, details="; ".join(syntax) or "ок")
    )
    if syntax:
        report.duration_s = time.monotonic() - started
        return report

    files = _test_files(spec)
    reference_run = run_tests(spec.reference_solution, files, timeout=timeout)
    report.reference_pass_rate = reference_run.pass_rate
    report.checks.append(
        ValidationCheck(
            name="Эталон проходит все тесты",
            ok=reference_run.all_passed,
            details=(
                f"{len(reference_run.passed_names)}/{len(reference_run.expected)} тестов"
                + ("" if reference_run.all_passed else " — " + _failed_summary(reference_run))
            ),
        )
    )

    if check_determinism and reference_run.all_passed:
        second = run_tests(spec.reference_solution, files, timeout=timeout)
        stable = second.passed_names == reference_run.passed_names
        report.checks.append(
            ValidationCheck(
                name="Тесты детерминированы",
                ok=stable,
                details="повторный прогон дал тот же результат" if stable else _failed_summary(second),
            )
        )

    starter_run = run_tests(spec.starter_code, files, timeout=timeout, per_test_timeout=5)
    report.starter_fail_rate = 1.0 - starter_run.pass_rate if starter_run.expected else 0.0
    hidden_names = set(test_names(spec.tests_hidden))
    starter_passes_hidden = bool(hidden_names) and hidden_names <= set(starter_run.passed_names)
    report.checks.append(
        ValidationCheck(
            name="Шаблон не решает задачу",
            ok=not starter_passes_hidden,
            details=f"шаблон проходит {len(starter_run.passed_names)}/{len(starter_run.expected)} тестов",
        )
    )
    if not starter_passes_hidden and starter_run.pass_rate > 0.5:
        report.checks.append(
            ValidationCheck(
                name="Шаблон проходит не больше половины тестов",
                ok=False,
                details=f"шаблон проходит {starter_run.pass_rate:.0%} тестов — задание может быть слишком простым",
                severity="warning",
            )
        )

    if run_mutants and reference_run.all_passed:
        if spec.mutants:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                results = list(pool.map(lambda m: _run_mutant(spec, m, files, timeout), spec.mutants))
            report.mutants = results
            not_applied = [m.id for m in results if not m.applied]
            if not_applied:
                report.checks.append(
                    ValidationCheck(
                        name="Мутанты применимы к эталону",
                        ok=False,
                        details="фрагмент не найден в эталоне: " + ", ".join(not_applied),
                        severity="warning",
                    )
                )
            score = report.mutation_score
            survived = [f"{m.id} ({m.description})" for m in results if m.applied and not m.killed]
            report.checks.append(
                ValidationCheck(
                    name="Тесты ловят типичные ошибки (мутационный анализ)",
                    ok=score is not None and score >= MIN_MUTATION_SCORE,
                    details=(
                        f"обнаружено {sum(m.killed for m in results if m.applied)} из "
                        f"{sum(m.applied for m in results)} мутантов"
                        + ("; выжили: " + ", ".join(survived) if survived else "")
                    ),
                )
            )
        else:
            report.checks.append(
                ValidationCheck(
                    name="Тесты ловят типичные ошибки (мутационный анализ)",
                    ok=False,
                    details="мутанты не заданы — качество тестов не проверено",
                    severity="warning",
                )
            )

    report.duration_s = time.monotonic() - started
    return report

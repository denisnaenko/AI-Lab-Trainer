"""Командная строка: labgen <команда>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import list_topics
from .config import load_settings
from .grader import grade_directory, grade_solution, report_to_markdown, reports_to_csv
from .levels import LEVELS
from .models import LabRecord, LabSpec
from .agent.graph import NODE_LABELS
from .llm import LLMError
from .pipeline.generator import TopicNotFound
from .service import EXPORT_KINDS, ApprovalError, LabService
from .storage import LabNotFound


def _service(args: argparse.Namespace) -> LabService:
    settings = load_settings()
    if getattr(args, "data", None):
        settings.data_dir = Path(args.data)
    return LabService(settings)


def _print_record(record: LabRecord) -> None:
    v = record.validation
    print(f"{record.id}  [{record.status}]  v{record.version}")
    print(f"  {record.spec.title} — уровень {record.level} ({LEVELS[record.level].name}), режим: {record.mode}")
    if v:
        mutation = f", мутанты {v.mutation_score:.0%}" if v.mutation_score is not None else ""
        print(f"  Проверка: {'пройдена' if v.passed else 'ЕСТЬ ОШИБКИ'}; покрытие {v.auto_coverage:.0%}{mutation}")
        for check in v.checks:
            if not check.ok:
                print(f"    {'!' if check.severity == 'error' else '~'} {check.name}: {check.details}")


def cmd_catalog(args: argparse.Namespace) -> int:
    for topic in list_topics():
        print(f"{topic.id:18} {topic.title}")
        print(f"{'':18} {topic.summary}")
    print("\nУровни:", "; ".join(f"{n} — {lvl.name}" for n, lvl in LEVELS.items()))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    service = _service(args)
    materials = Path(args.materials).read_text(encoding="utf-8") if args.materials else None
    try:
        record = service.create(
            args.topic, args.level, mode=args.mode, llm=service.default_llm, materials=materials,
            topic_id=args.topic_id, simulate=args.simulate,
            on_step=lambda node, log: print(f"  → {NODE_LABELS.get(node, node)}", flush=True),
        )
    except (TopicNotFound, ValueError, LLMError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    _print_record(record)
    for line in record.generation_log:
        print("  ·", line)
    return 0 if record.validation and record.validation.passed else 1


def cmd_list(args: argparse.Namespace) -> int:
    records = _service(args).store.list()
    if not records:
        print("Лабораторных работ пока нет. Создайте: labgen generate \"Агент с инструментами\" -l 1")
    for r in records:
        ok = "✓" if r.validation and r.validation.passed else "✗"
        print(f"{ok} {r.id:45} {r.status:9} L{r.level}  {r.spec.title}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    record = _service(args).store.get(args.lab_id)
    _print_record(record)
    print()
    print(record.spec.statement_md)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    record = _service(args).validate(args.lab_id)
    _print_record(record)
    return 0 if record.validation and record.validation.passed else 1


def cmd_approve(args: argparse.Namespace) -> int:
    try:
        record = _service(args).approve(args.lab_id)
    except ApprovalError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    _print_record(record)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    service = _service(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = list(EXPORT_KINDS) if args.kind == "all" else [args.kind]
    for kind in kinds:
        filename, data = service.export(args.lab_id, kind)
        (out / filename).write_bytes(data)
        print(f"{EXPORT_KINDS[kind]}: {out / filename}")
    return 0


def _load_spec(service: LabService, target: str) -> tuple[LabSpec, str | None]:
    path = Path(target)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if "spec" in data:
            return LabRecord.model_validate(data).spec, None
        return LabSpec.model_validate(data), None
    record = service.store.get(target)
    return record.spec, record.id


def cmd_grade(args: argparse.Namespace) -> int:
    service = _service(args)
    spec, lab_id = _load_spec(service, args.lab)
    llm = service.default_llm if args.llm_feedback else None
    target = Path(args.solution)
    if target.is_dir():
        reports = grade_directory(spec, target, llm=llm)
        csv_text = reports_to_csv(reports)
        if args.csv:
            Path(args.csv).write_text(csv_text, encoding="utf-8-sig")
            print(f"Результаты {len(reports)} решений: {args.csv}")
        else:
            print(csv_text)
        if lab_id:
            for report in reports:
                service.store.save_grade(lab_id, report)
        return 0
    code = target.read_text(encoding="utf-8", errors="replace")
    report = grade_solution(spec, code, student=args.student or target.stem, llm=llm)
    if lab_id:
        service.store.save_grade(lab_id, report)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(report_to_markdown(report))
    return 0


def cmd_build_catalog(args: argparse.Namespace) -> int:
    """Создаёт все работы каталога (5 тем × 3 уровня), проверяет и экспортирует их."""
    service = _service(args)
    out = Path(args.out)
    failures = 0
    for topic in list_topics():
        for level in args.levels:
            record = service.create(topic.title, level, mode="catalog", topic_id=topic.id, simulate=0)
            v = record.validation
            passed = bool(v and v.passed)
            if passed and args.approve:
                record = service.approve(record.id)
            failures += 0 if passed else 1
            folder = out / record.id
            folder.mkdir(parents=True, exist_ok=True)
            for kind in EXPORT_KINDS:
                filename, data = service.export(record.id, kind)
                (folder / filename).write_bytes(data)
            mutation = f"{v.mutation_score:.0%}" if v and v.mutation_score is not None else "—"
            print(
                f"{'✓' if passed else '✗'} {topic.title} — уровень {level}: покрытие {v.auto_coverage:.0%}, "
                f"мутанты {mutation} → {folder}"
            )
    return 1 if failures else 0


def cmd_graph(args: argparse.Namespace) -> int:
    from .agent.graph import LAB_AGENT

    print(LAB_AGENT.get_graph().draw_mermaid())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web.app import create_app

    settings = load_settings()
    if args.data:
        settings.data_dir = Path(args.data)
    uvicorn.run(create_app(LabService(settings)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labgen", description="ИИ-агент для создания лабораторных работ по агентной разработке")
    parser.add_argument("--data", help="каталог данных (по умолчанию LABGEN_DATA_DIR или ./data)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("catalog", help="темы встроенного каталога").set_defaults(func=cmd_catalog)

    p = sub.add_parser("generate", help="создать лабораторную по теме")
    p.add_argument("topic", help="тема занятия, например «Агент с инструментами»")
    p.add_argument("-l", "--level", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--mode", choices=["auto", "catalog", "llm"], default="auto")
    p.add_argument("--topic-id", help="тема каталога (structured_output, tool_agent, ...)")
    p.add_argument("--materials", help="файл с материалами преподавателя (txt/md) для LLM-режима")
    p.add_argument("--simulate", type=int, default=None, help="число симулированных студентов (LLM-режим)")
    p.set_defaults(func=cmd_generate)

    sub.add_parser("list", help="список лабораторных").set_defaults(func=cmd_list)
    for name, func, text in (
        ("show", cmd_show, "показать работу"),
        ("validate", cmd_validate, "перепроверить работу"),
        ("approve", cmd_approve, "утвердить к публикации"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("lab_id")
        p.set_defaults(func=func)

    p = sub.add_parser("export", help="экспорт в Gitverse / Moodle / комплект преподавателя")
    p.add_argument("lab_id")
    p.add_argument("--kind", choices=[*EXPORT_KINDS, "all"], default="all")
    p.add_argument("--out", default="export")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("grade", help="автопроверка решения или папки решений")
    p.add_argument("lab", help="id работы или путь к lab.json из комплекта преподавателя")
    p.add_argument("solution", help="файл solution.py или папка с решениями")
    p.add_argument("--student")
    p.add_argument("--csv", help="куда сохранить таблицу (для папки)")
    p.add_argument("--json", action="store_true", help="вывести отчёт в JSON")
    p.add_argument("--llm-feedback", action="store_true", help="пояснения ошибок от LLM (баллы не меняются)")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("build-catalog", help="создать и экспортировать все работы каталога")
    p.add_argument("--out", default="export/catalog")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--approve", action="store_true", help="сразу утвердить прошедшие проверку")
    p.set_defaults(func=cmd_build_catalog)

    p = sub.add_parser("graph", help="схема графа агента (Mermaid)")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("serve", help="веб-интерфейс преподавателя")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LabNotFound as exc:
        print(f"Ошибка: работа {exc} не найдена", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

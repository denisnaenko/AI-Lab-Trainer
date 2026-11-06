"""Описание темы каталога и сборка лабораторной работы нужного уровня.

Каждая тема хранит исходники в ``sources/<topic_id>/``:

* ``reference.py`` — эталонное решение (покрывает все уровни);
* ``starter_1.py`` … ``starter_3.py`` — шаблоны для студентов;
* ``tests.py`` — все тесты темы. Сначала идут импорты и помощники, затем
  тестовые функции ``test_r<N>_...``; уровень теста = уровень требования R<N>.

Метаданные (требования, мутанты, подсказки) описаны в модуле темы.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..levels import get_level
from ..models import LabSpec, Mutant, Requirement, requirement_of

SOURCES_DIR = Path(__file__).parent / "sources"

MANUAL_REQUIREMENT = (
    "Код читаемый: понятные имена, docstring у публичных функций и классов, нет дублирования; "
    "решение не обращается к сети и файловой системе. Оценивает преподаватель."
)
MANUAL_POINTS = 10


@dataclass(frozen=True)
class RequirementDef:
    id: str
    level: int
    text: str


@dataclass(frozen=True)
class MutantDef:
    id: str
    level: int
    description: str
    find: str
    replace: str


@dataclass
class TopicDef:
    id: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    context_md: str
    interface_md: dict[int, str]
    requirements: list[RequirementDef]
    public_tests: set[str]
    mutants: list[MutantDef]
    hints: dict[int, list[str]] = field(default_factory=dict)
    minutes: dict[int, int] = field(default_factory=lambda: {1: 45, 2: 90, 3: 135})

    # --- исходники ------------------------------------------------------------

    def source(self, name: str) -> str:
        return (SOURCES_DIR / self.id / name).read_text(encoding="utf-8")

    @property
    def reference(self) -> str:
        return self.source("reference.py")

    def starter(self, level: int) -> str:
        return self.source(f"starter_{level}.py")

    def _split_tests(self) -> tuple[str, list[tuple[str, str]]]:
        code = self.source("tests.py")
        lines = code.splitlines()
        tree = ast.parse(code)
        tests: list[tuple[str, str]] = []
        in_test = [False] * len(lines)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                tests.append((node.name, "\n".join(lines[start - 1 : node.end_lineno])))
                for i in range(start - 1, node.end_lineno):
                    in_test[i] = True
        # всё, что не тесты (импорты, константы, помощники), — общий заголовок файла
        header_lines = [line for line, is_test in zip(lines, in_test) if not is_test]
        header = re.sub(r"\n{4,}", "\n\n\n", "\n".join(header_lines)).strip()
        return header, tests

    def requirement_level(self, req_id: str) -> int:
        for r in self.requirements:
            if r.id == req_id:
                return r.level
        raise KeyError(f"{self.id}: тест ссылается на неизвестное требование {req_id}")

    # --- сборка ---------------------------------------------------------------

    def build(self, level: int) -> LabSpec:
        lvl = get_level(level)
        auto_reqs = [r for r in self.requirements if r.level <= level]
        auto_points = 100 - MANUAL_POINTS
        base, extra = divmod(auto_points, len(auto_reqs))
        requirements = [
            Requirement(id=r.id, text=r.text, points=base + (1 if i < extra else 0), auto=True)
            for i, r in enumerate(auto_reqs)
        ]
        manual_id = f"R{len(auto_reqs) + 1}"
        requirements.append(Requirement(id=manual_id, text=MANUAL_REQUIREMENT, points=MANUAL_POINTS, auto=False))

        header, tests = self._split_tests()
        public, hidden = [], []
        for name, code in tests:
            req = requirement_of(name)
            if req is None or self.requirement_level(req) > level:
                continue
            (public if name in self.public_tests else hidden).append(code)

        def assemble(parts: list[str], kind: str) -> str:
            doc = f'"""{kind} тесты: {self.title} (уровень {level})."""\n'
            return doc + header + "\n\n\n" + "\n\n\n".join(parts) + "\n"

        mutants = [
            Mutant(id=m.id, description=m.description, find=m.find, replace=m.replace)
            for m in self.mutants
            if m.level <= level
        ]
        spec = LabSpec(
            topic_id=self.id,
            title=f"{self.title}",
            level=level,
            summary=self.summary,
            statement_md="",
            starter_code=self.starter(level),
            reference_solution=self.reference,
            tests_public=assemble(public, "Публичные"),
            tests_hidden=assemble(hidden, "Скрытые"),
            requirements=requirements,
            mutants=mutants,
            estimated_minutes=self.minutes.get(level),
        )
        spec.statement_md = render_statement(
            spec,
            context_md=self.context_md,
            interface_md=self.interface_md.get(level, ""),
            hints=self.hints.get(level, []),
            level_name=lvl.name,
        )
        return spec


def render_statement(
    spec: LabSpec,
    *,
    context_md: str,
    interface_md: str,
    hints: list[str],
    level_name: str,
) -> str:
    """Текст задания для студента (Markdown)."""
    parts = [f"# {spec.title}", ""]
    meta = f"**Уровень сложности:** {spec.level} — {level_name}."
    if spec.estimated_minutes:
        meta += f" **Ориентировочное время:** {spec.estimated_minutes} мин."
    parts += [meta, "", "## Контекст", "", context_md.strip(), ""]
    parts += [
        "## Что нужно сделать",
        "",
        interface_md.strip(),
        "",
        "Решение пишется в файле `solution.py` — шаблон с интерфейсом уже лежит в репозитории. "
        "Не меняйте имена функций, классов и параметров: по ним работает автопроверка.",
        "",
    ]
    parts += ["### Требования", ""]
    for r in spec.requirements:
        mark = "" if r.auto else " *(проверяет преподаватель)*"
        parts.append(f"- **{r.id}.** {r.text}{mark}")
    parts.append("")
    if hints:
        parts += ["## Подсказки", ""] + [f"- {h}" for h in hints] + [""]
    parts += [
        "## Как проверить себя",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "pytest",
        "```",
        "",
        "В тестах настоящая модель заменена на `ScriptedLLM` из файла `agentkit.py`: "
        "она возвращает заранее заданные ответы и запоминает все запросы вашего агента. "
        "Поэтому тесты проверяют не текст ответа модели, а поведение агента: какие сообщения "
        "он отправил, какие инструменты вызвал и когда остановился.",
        "",
        "Публичные тесты в `tests/test_public.py` проверяют часть требований. "
        "При сдаче решение проверяется полным набором скрытых тестов.",
        "",
        "## Оценивание",
        "",
        "| Требование | Баллы | Проверка |",
        "| --- | --- | --- |",
    ]
    for r in spec.requirements:
        parts.append(f"| {r.id} | {r.points} | {'автотесты' if r.auto else 'преподаватель'} |")
    parts += ["", f"Всего: {spec.total_points} баллов."]
    return "\n".join(parts) + "\n"

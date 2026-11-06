"""Экспорт в Moodle.

1. ``moodle_coderunner.xml`` — вопрос типа CodeRunner (плагин qtype_coderunner) в формате
   Moodle XML: импорт через «Банк вопросов → Импорт → Moodle XML». Студент вставляет код
   solution.py, CodeRunner прогоняет каждый тест и начисляет баллы требования.
   Нужны установленные плагин CodeRunner и сервер Jobe.
2. ``assignment.html`` — описание для элемента курса «Задание» (сдача ссылкой на репозиторий).

Как работает шаблон вопроса: модуль agentkit встраивается в шаблон в base64, ответ
студента подставляется через ``{{ STUDENT_ANSWER | e('py') }}`` и загружается как модуль
``solution``, после чего выполняется код теста (``{{ TEST.testcode }}``). Тест печатает
``OK``, если проверка прошла, иначе — причину ошибки.
"""

from __future__ import annotations

import base64
import html
from xml.sax.saxutils import escape

import markdown

from ..models import LabSpec, requirement_of
from ..sandbox import AGENTKIT_SOURCE
from . import lab_slug, make_zip


def statement_html(spec: LabSpec) -> str:
    return markdown.markdown(spec.statement_md, extensions=["tables", "fenced_code"])


def coderunner_template() -> str:
    agentkit_b64 = base64.b64encode(AGENTKIT_SOURCE.encode("utf-8")).decode("ascii")
    return (
        "import base64\n"
        "import sys\n"
        "import types\n"
        "\n"
        "\n"
        "def _load_module(name, source):\n"
        "    module = types.ModuleType(name)\n"
        "    module.__file__ = name + '.py'\n"
        "    sys.modules[name] = module\n"
        "    exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    return module\n"
        "\n"
        "\n"
        f"_load_module('agentkit', base64.b64decode('{agentkit_b64}').decode('utf-8'))\n"
        "__student_answer__ = \"\"\"{{ STUDENT_ANSWER | e('py') }}\"\"\"\n"
        "try:\n"
        "    _load_module('solution', __student_answer__)\n"
        "except Exception as _exc:\n"
        "    print('Решение не загрузилось:', type(_exc).__name__, _exc)\n"
        "    sys.exit(0)\n"
        "\n"
        "{{ TEST.testcode }}\n"
    )


def _test_header(module_code: str) -> str:
    """Импорты и помощники тестового модуля (всё до первой тестовой функции)."""
    import ast

    tree = ast.parse(module_code)
    lines = module_code.splitlines()
    keep = [True] * len(lines)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            for i in range(start - 1, node.end_lineno):
                keep[i] = False
    return "\n".join(line for line, k in zip(lines, keep) if k).strip()


def _test_functions(module_code: str) -> list[tuple[str, str]]:
    import ast

    tree = ast.parse(module_code)
    lines = module_code.splitlines()
    result = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result.append((node.name, "\n".join(lines[start - 1 : node.end_lineno])))
    return result


def testcase_code(header: str, function_code: str, name: str) -> str:
    return (
        f"{header}\n\n\n{function_code}\n\n\n"
        "try:\n"
        f"    {name}()\n"
        "    print('OK')\n"
        "except AssertionError as _error:\n"
        "    print('Не пройдено:', _error)\n"
        "except Exception as _error:\n"
        "    print('Ошибка:', type(_error).__name__, _error)\n"
    )


def build_testcases(spec: LabSpec) -> list[dict]:
    mapping = spec.tests_by_requirement()
    cases = []
    for code, public in ((spec.tests_public, True), (spec.tests_hidden, False)):
        header = _test_header(code)
        for name, function_code in _test_functions(code):
            req = spec.requirement(requirement_of(name) or "")
            if req is None or not req.auto:
                continue
            mark = req.points / max(1, len(mapping.get(req.id, [])))
            cases.append(
                {
                    "name": name,
                    "requirement": req.id,
                    "public": public,
                    "mark": mark,
                    "code": testcase_code(header, function_code, name),
                }
            )
    return cases


def _cdata(text: str) -> str:
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def coderunner_xml(spec: LabSpec, category: str = "AI Lab Trainer") -> str:
    cases = build_testcases(spec)
    total = sum(c["mark"] for c in cases)
    testcases_xml = []
    for case in cases:
        testcases_xml.append(
            f'      <testcase testtype="0" useasexample="{1 if case["public"] else 0}" '
            f'hiderestiffail="0" mark="{case["mark"]:.7f}">\n'
            f"        <testcode><text>{_cdata(case['code'])}</text></testcode>\n"
            "        <stdin><text></text></stdin>\n"
            "        <expected><text>OK</text></expected>\n"
            "        <extra><text></text></extra>\n"
            f"        <display><text>{'SHOW' if case['public'] else 'HIDE'}</text></display>\n"
            "      </testcase>"
        )
    question_html = statement_html(spec) + (
        "<p><em>Вставьте полный код файла solution.py. Проверка: "
        f"{len(cases)} тестов, видимые — публичные.</em></p>"
    )
    title = f"{spec.title} (уровень {spec.level})"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<quiz>
  <question type="category">
    <category><text>{escape("$course$/top/" + category)}</text></category>
  </question>
  <question type="coderunner">
    <name><text>{escape(title)}</text></name>
    <questiontext format="html"><text>{_cdata(question_html)}</text></questiontext>
    <generalfeedback format="html"><text></text></generalfeedback>
    <defaultgrade>{total:.7f}</defaultgrade>
    <penalty>0.0000000</penalty>
    <hidden>0</hidden>
    <coderunnertype>python3</coderunnertype>
    <prototypetype>0</prototypetype>
    <allornothing>0</allornothing>
    <penaltyregime>0</penaltyregime>
    <precheck>2</precheck>
    <showsource>0</showsource>
    <answerboxlines>30</answerboxlines>
    <answerboxcolumns>100</answerboxcolumns>
    <answerpreload>{_cdata(spec.starter_code)}</answerpreload>
    <useace>1</useace>
    <resultcolumns></resultcolumns>
    <template>{_cdata(coderunner_template())}</template>
    <iscombinatortemplate>0</iscombinatortemplate>
    <allowmultiplestdins>0</allowmultiplestdins>
    <answer>{_cdata(spec.reference_solution)}</answer>
    <validateonsave>1</validateonsave>
    <testsplitterre></testsplitterre>
    <language>python3</language>
    <acelang>python</acelang>
    <sandbox></sandbox>
    <grader>EqualityGrader</grader>
    <cputimelimitsecs>10</cputimelimitsecs>
    <memlimitmb>500</memlimitmb>
    <sandboxparams></sandboxparams>
    <templateparams></templateparams>
    <twigall>0</twigall>
    <testcases>
{chr(10).join(testcases_xml)}
    </testcases>
  </question>
</quiz>
"""


def assignment_html(spec: LabSpec) -> str:
    return (
        f"<h2>{html.escape(spec.title)} (уровень {spec.level})</h2>\n"
        + statement_html(spec)
        + "\n<h3>Сдача</h3>\n<p>Выполните работу в выданном репозитории и прикрепите ссылку на него "
        "в ответе на задание. Итоговую оценку выставляет автопроверка по скрытым тестам и преподаватель.</p>\n"
    )


MOODLE_README = """# Импорт в Moodle: {title} (уровень {level})

## Вариант 1. Вопрос CodeRunner (автопроверка в Moodle)

Требуется плагин CodeRunner (qtype_coderunner) и сервер Jobe.

1. Курс → Банк вопросов → Импорт → формат «Moodle XML» → файл `moodle_coderunner.xml`.
2. Добавьте вопрос в тест (Quiz). Студент вставляет полный код `solution.py`.
3. Публичные тесты видны студенту, скрытые — нет. Баллы теста = баллы требования / число его тестов.
4. Требование о качестве кода оценивается вручную (комментарий к попытке).

После импорта откройте вопрос и сохраните его: при `validateonsave` CodeRunner прогонит эталонное
решение на всех тестах — так вы убедитесь, что вопрос работает на вашем сервере Jobe.

## Вариант 2. Элемент «Задание»

Создайте «Задание», вставьте текст из `assignment.html`, выдайте студентам репозиторий
(архив `*-student.zip`) и проверяйте решения командой `labgen grade`.
"""


def export_moodle(spec: LabSpec) -> tuple[str, bytes]:
    slug = lab_slug(spec)
    files = {
        "moodle_coderunner.xml": coderunner_xml(spec),
        "assignment.html": assignment_html(spec),
        "README.md": MOODLE_README.format(title=spec.title, level=spec.level),
    }
    return f"{slug}-moodle.zip", make_zip(files, root=f"{slug}-moodle")

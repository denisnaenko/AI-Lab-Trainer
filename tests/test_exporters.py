import io
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest

from labgen.catalog import build_lab
from labgen.exporters.gitverse import export_student_repo
from labgen.exporters.moodle import build_testcases, coderunner_template, coderunner_xml, export_moodle
from labgen.exporters.teacher import export_teacher_kit


def unzip(data: bytes, target):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(target)
        return archive.namelist()


def run_pytest(folder):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=folder, capture_output=True, text=True)


def test_student_repo_runs_public_tests(tmp_path):
    spec = build_lab("tool_agent", 2)
    name, data = export_student_repo(spec)
    names = unzip(data, tmp_path)
    root = tmp_path / name.removesuffix("-student.zip")
    assert {"README.md", "solution.py", "agentkit.py", "tests/test_public.py", ".gitverse/workflows/tests.yaml"} <= {
        n.split("/", 1)[1] for n in names
    }
    assert "test_hidden" not in data.decode("latin-1"), "скрытые тесты не должны попадать к студентам"
    assert run_pytest(root).returncode != 0, "шаблон не должен проходить тесты"
    (root / "solution.py").write_text(spec.reference_solution, encoding="utf-8")
    result = run_pytest(root)
    assert result.returncode == 0, result.stdout


def test_teacher_kit_contains_hidden_tests_and_rubric(tmp_path):
    spec = build_lab("rag_agent", 3)
    name, data = export_teacher_kit(spec)
    names = unzip(data, tmp_path)
    assert any(n.endswith("tests/test_hidden.py") for n in names)
    assert any(n.endswith("reference/solution.py") for n in names)
    rubric = next(p for p in tmp_path.rglob("RUBRIC.md")).read_text(encoding="utf-8")
    assert "R8" in rubric and "test_r8_" in rubric


def py_escape(text: str) -> str:
    """Эмуляция фильтра CodeRunner e('py') для строк в кавычках Python."""
    return text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')


def run_coderunner_case(answer: str, testcode: str) -> str:
    program = coderunner_template().replace("{{ STUDENT_ANSWER | e('py') }}", py_escape(answer)).replace(
        "{{ TEST.testcode }}", testcode
    )
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30).stdout.strip()


@pytest.mark.parametrize("topic_id", ["structured_output", "multi_agent"])
def test_moodle_coderunner_question_logic(topic_id):
    spec = build_lab(topic_id, 2)
    cases = build_testcases(spec)
    assert abs(sum(c["mark"] for c in cases) - 90) < 1e-6
    assert all(run_coderunner_case(spec.reference_solution, c["code"]) == "OK" for c in cases)
    assert not any(run_coderunner_case(spec.starter_code, c["code"]) == "OK" for c in cases)


def test_moodle_xml_is_well_formed():
    spec = build_lab("memory_agent", 3)
    root = ET.fromstring(coderunner_xml(spec))
    question = root.find("question[@type='coderunner']")
    assert question.find("coderunnertype").text == "python3"
    testcases = question.find("testcases")
    assert len(testcases) == len(build_testcases(spec))
    shown = [tc for tc in testcases if tc.find("display/text").text == "SHOW"]
    assert shown and len(shown) < len(testcases)
    template = question.find("template").text
    assert template.count("{{") == 2, "в шаблоне не должно быть лишних конструкций Twig"
    name, data = export_moodle(spec)
    assert name.endswith("-moodle.zip") and len(data) > 1000

"""Изолированный запуск тестов для решения студента или эталона.

MVP-уровень изоляции: отдельный процесс и временный каталог, лимиты времени,
CPU и памяти, урезанное окружение, запрет сетевых подключений и тайм-аут на
каждый тест. Для промышленной эксплуатации запуск нужно перенести в контейнер
без сети (Docker/gVisor) — интерфейс ``run_tests`` для этого менять не придётся.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .models import test_names

RUNTIME_DIR = Path(__file__).parent / "runtime"
AGENTKIT_SOURCE = (RUNTIME_DIR / "agentkit.py").read_text(encoding="utf-8")

CONFTEST_SOURCE = '''"""Ограничения окружения для автопроверки (создаётся автоматически)."""
import signal
import socket

import pytest

PER_TEST_TIMEOUT = {per_test_timeout}


def _blocked(*args, **kwargs):
    raise RuntimeError("Сетевые подключения в автопроверке запрещены")


socket.socket.connect = _blocked
socket.create_connection = _blocked


def _on_timeout(signum, frame):
    raise TimeoutError(f"Тест выполняется дольше {{PER_TEST_TIMEOUT}} с — возможно, бесконечный цикл")


@pytest.fixture(autouse=True)
def _per_test_timeout():
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(PER_TEST_TIMEOUT)
        yield
        signal.alarm(0)
    else:
        yield
'''


@dataclass
class TestOutcome:
    name: str
    status: str  # passed | failed | error | skipped | missing
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class RunResult:
    expected: list[str]
    outcomes: dict[str, TestOutcome] = field(default_factory=dict)
    returncode: int | None = None
    timed_out: bool = False
    output: str = ""
    duration_s: float = 0.0
    collection_error: str = ""

    def outcome(self, name: str) -> TestOutcome:
        return self.outcomes.get(name) or TestOutcome(name, "missing", self.collection_error or "тест не запустился")

    @property
    def passed_names(self) -> list[str]:
        return [n for n in self.expected if self.outcome(n).passed]

    @property
    def failed_names(self) -> list[str]:
        return [n for n in self.expected if not self.outcome(n).passed]

    @property
    def pass_rate(self) -> float:
        return len(self.passed_names) / len(self.expected) if self.expected else 0.0

    @property
    def all_passed(self) -> bool:
        return bool(self.expected) and not self.failed_names


# не давать pytest собирать эти классы как тесты
TestOutcome.__test__ = False  # type: ignore[attr-defined]


def _limit_resources(cpu_seconds: int) -> None:  # pragma: no cover - выполняется в дочернем процессе
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 * 1024, 1536 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    except Exception:
        pass


def _parse_junit(path: Path, expected: list[str]) -> tuple[dict[str, TestOutcome], str]:
    outcomes: dict[str, TestOutcome] = {}
    collection_error = ""
    if not path.exists():
        return outcomes, collection_error
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        name = case.get("name", "")
        status, message = "passed", ""
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                message = (child.get("message") or "") + "\n" + (child.text or "")
                message = message.strip()
                break
        if name not in expected:
            # ошибка сбора модуля (например, синтаксическая ошибка в решении)
            if status == "passed":
                continue
            collection_error = collection_error or message
            continue
        outcomes[name] = TestOutcome(name, status, _shorten(message))
    return outcomes, _tail(collection_error, 1200)


def _tail(text: str, limit: int) -> str:
    """Хвост сообщения: в трассировке ошибки сбора главное — последние строки."""
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def _shorten(text: str, limit: int = 800) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_tests(
    solution_code: str,
    test_files: dict[str, str],
    *,
    timeout: float = 60.0,
    per_test_timeout: int = 10,
    extra_files: dict[str, str] | None = None,
) -> RunResult:
    """Запускает тесты ``test_files`` против решения ``solution_code``.

    ``test_files`` — словарь «имя файла → код», например ``{"test_hidden.py": "..."}``.
    Решение кладётся в ``solution.py``, рядом — ``agentkit.py``.
    """
    expected: list[str] = []
    for code in test_files.values():
        expected.extend(test_names(code))

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="labgen_") as tmp:
        workdir = Path(tmp)
        (workdir / "solution.py").write_text(solution_code, encoding="utf-8")
        (workdir / "agentkit.py").write_text(AGENTKIT_SOURCE, encoding="utf-8")
        (workdir / "conftest.py").write_text(
            CONFTEST_SOURCE.format(per_test_timeout=int(per_test_timeout)), encoding="utf-8"
        )
        for name, content in (extra_files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")
        for name, code in test_files.items():
            (workdir / name).write_text(code, encoding="utf-8")

        report = workdir / "report.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(workdir),
            f"--junitxml={report}",
            *test_files.keys(),
        ]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(workdir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "HOME": str(workdir),
            "LANG": "C.UTF-8",
        }
        preexec = None
        if os.name == "posix":
            cpu = int(timeout) + 5
            preexec = lambda: _limit_resources(cpu)  # noqa: E731

        result = RunResult(expected=expected)
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
            )
            result.returncode = completed.returncode
            result.output = _shorten(completed.stdout + completed.stderr, 6000)
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            out = exc.stdout or ""
            result.output = _shorten(out if isinstance(out, str) else out.decode("utf-8", "replace"), 6000)
        outcomes, collection_error = _parse_junit(report, expected)
        result.outcomes = outcomes
        result.collection_error = collection_error
        if result.timed_out and not outcomes:
            result.collection_error = f"Превышено общее время проверки ({timeout:.0f} с)"
        elif not outcomes and not collection_error and result.returncode not in (0, None):
            result.collection_error = _shorten(result.output, 1500) or "тесты не запустились"
    result.duration_s = time.monotonic() - started
    return result

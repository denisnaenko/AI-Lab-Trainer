"""Репозиторий-шаблон для студента (Gitverse или любой git-хостинг).

Содержит условие, шаблон решения, среду agentkit, публичные тесты и CI, который
запускает публичные тесты при каждом push. Скрытые тесты в репозиторий студента
не попадают — они в комплекте преподавателя.

CI-файл лежит в ``.gitverse/workflows/`` и использует синтаксис GitHub Actions
(``on``/``jobs``/``runs-on``/``steps``), который применяется в GitVerse CI/CD.
Перед первым запуском курса проверьте его на тестовом репозитории.
"""

from __future__ import annotations

from ..models import LabSpec
from ..sandbox import AGENTKIT_SOURCE
from . import lab_slug, make_zip

WORKFLOW = """name: Автопроверка (публичные тесты)

on:
  push:
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Установка зависимостей
        run: python3 -m pip install -r requirements.txt
      - name: Публичные тесты
        run: python3 -m pytest -v
"""

PYTEST_INI = """[pytest]
pythonpath = .
testpaths = tests
"""

GITIGNORE = """__pycache__/
*.pyc
.pytest_cache/
.venv/
.env
"""

SUBMIT_NOTE = """
## Как сдать работу

1. Решение — в файле `solution.py`. Не переименовывайте функции, классы и параметры из шаблона.
2. Сделайте commit и push в этот репозиторий: CI запустит публичные тесты (вкладка CI/CD).
3. Окончательную оценку выставит автопроверка преподавателя по полному набору тестов.

Для экспериментов с настоящей моделью в `agentkit.py` есть клиент `OpenAICompatibleLLM`
(настройки — переменные окружения `AGENTKIT_BASE_URL`, `AGENTKIT_API_KEY`, `AGENTKIT_MODEL`).
В тестах он не используется.
"""


def student_repo_files(spec: LabSpec) -> dict[str, str]:
    return {
        "README.md": spec.statement_md.rstrip() + "\n" + SUBMIT_NOTE,
        "solution.py": spec.starter_code,
        "agentkit.py": AGENTKIT_SOURCE,
        "tests/test_public.py": spec.tests_public,
        "requirements.txt": "pytest>=8.0\n",
        "pytest.ini": PYTEST_INI,
        ".gitignore": GITIGNORE,
        ".gitverse/workflows/tests.yaml": WORKFLOW,
    }


def export_student_repo(spec: LabSpec) -> tuple[str, bytes]:
    slug = lab_slug(spec)
    return f"{slug}-student.zip", make_zip(student_repo_files(spec), root=slug)

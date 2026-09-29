"""Prompt templates, one per graph node. Kept separate so methodologists can edit them
without touching code. Variables in {braces} are filled by `graph.py`."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """\
Ты методист и преподаватель курса по разработке ИИ-агентов. Ты проектируешь
лабораторные работы на Python, которые проверяются автоматически через pytest.

Соглашения, которые нельзя нарушать:
- Код студентов строится на LangChain / LangGraph (langchain-core, langgraph): инструменты
  через @tool, сообщения HumanMessage / AIMessage / ToolMessage, модель передаётся в код
  студента аргументом (объект чат-модели с bind_tools и invoke).
- Тесты никогда не обращаются к реальной LLM: tests/conftest.py содержит фикстуру fake_llm —
  чат-модель LangChain, которая возвращает заранее заданные AIMessage (текст или tool_calls)
  и записывает, какие сообщения ей передали.
- id теста — это имя pytest-функции; file — путь внутри tests/ (например test_agent.py).
- Эталонное решение проходит все тесты; шаблон (starter) не проходит ни одного
  автопроверяемого теста: недостающие места помечены TODO и бросают NotImplementedError.
"""

DIFFICULTY_GUIDE = {
    "basic": "Почти весь код дан, студент заполняет помеченные TODO. Все тесты открыты.",
    "intermediate": "Дан каркас и интерфейсы, логику пишет студент. Часть тестов скрыта.",
    "advanced": "Дан только контракт публичного API. Большая часть тестов скрыта.",
}

PLAN = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", """\
Тема занятия: {topic}
Уровень сложности: {difficulty} ({difficulty_guide})
Язык материалов: {language}
Пожелания преподавателя: {teacher_notes}

Составь план лабораторной: id (латиница, цифры и дефисы), название, цели обучения,
оценку времени и 5-10 проверяемых требований с id R1, R2, ...
Не менее 80% требований должны проверяться автоматически (check=auto).
Поле test_ids оставь пустым: тесты появятся на следующем шаге."""),
])

MATERIALS = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", """\
Уровень сложности: {difficulty} ({difficulty_guide})
План лабораторной:
{plan}

Напиши материалы:
- brief: цель и контекст (Markdown);
- instructions: пошаговая инструкция для студента (Markdown);
- starter_files: шаблон решения по уровню сложности, включая requirements.txt;
- solution_files: эталонное решение с теми же путями модулей, что и в шаблоне."""),
])

TESTS = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", """\
Уровень сложности: {difficulty} ({difficulty_guide})
План лабораторной:
{plan}

Эталонное решение:
{solution}

Шаблон решения:
{starter}

Напиши автотесты:
- test_files: pytest-файлы и conftest.py с фикстурой fake_llm;
- tests: индекс тестов; у каждого автопроверяемого требования хотя бы один тест,
  hidden=true для скрытых тестов по уровню сложности;
- rubric: по одному пункту на каждое требование, включая ручные."""),
])

REPAIR = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", """\
Лабораторная не прошла проверку.

Ошибки:
{errors}

Вывод pytest:
{test_output}

План:
{plan}

Текущие материалы:
{materials}

Текущие тесты:
{suite}

Исправь лабораторную. Верни plan, materials и/или suite целиком в исправленном виде;
то, что исправлять не нужно, оставь пустым (null). План меняй, только если без этого
не добиться автопроверки 80% требований."""),
])

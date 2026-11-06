"""ИИ-агент преподавателя — граф LangGraph.

::

    START → plan ─┬─ catalog ──────────────┐
                  └─ generate ⇄ (ошибка разбора)
                        │                  ▼
                        └────────────→ validate ─┬─ (ошибки, есть попытки) → generate
                                                 ├─ (ошибки, попытки кончились) → finalize
                                                 └─ (пройдена) ──────────────┐
                                                         finalize → review → simulate → END

* **plan** — выбирает режим (каталог или LLM) и образец для few-shot;
* **catalog** — берёт проверенную работу из каталога;
* **generate** — LLM создаёт работу или исправляет её по замечаниям валидатора;
* **validate** — песочница: эталон, шаблон, мутанты, покрытие автопроверкой;
* **finalize** — если попытки кончились, оставляет лучшую версию как черновик;
* **review** — замечания для преподавателя (правила + LLM), без баллов;
* **simulate** — симулированные студенты решают задание (по желанию).

Модель передаётся через ``config["configurable"]["llm"]`` (объект :class:`LabLLM`
поверх чат-модели LangChain), поэтому один скомпилированный граф обслуживает
всех пользователей с их собственными ключами.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..levels import get_level
from ..llm.factory import LabLLM, LLMError
from ..models import LabSpec, ReviewRemark, ValidationReport
from ..pipeline import prompts
from ..pipeline.generator import (
    TopicNotFound,
    catalog_spec,
    parse_error_text,
    parse_spec,
    pick_example,
    quality,
    report_line,
    validation_issues,
)
from ..pipeline.reviewer import review_spec
from ..pipeline.students import simulate_students
from ..pipeline.validator import validate_spec

Mode = Literal["auto", "catalog", "llm"]

NODE_LABELS = {
    "plan": "Планирование",
    "catalog": "Выбор работы из каталога",
    "generate": "Генерация задания (LLM)",
    "validate": "Проверка в песочнице",
    "finalize": "Выбор лучшей версии",
    "review": "Рецензия",
    "simulate": "Симуляция студентов",
}


class LabState(TypedDict, total=False):
    # вход
    request: str
    level: int
    mode: str
    topic_id: str | None
    materials: str | None
    max_attempts: int
    simulate: int
    # рабочее состояние
    example: LabSpec | None
    raw: str
    spec: LabSpec | None
    report: ValidationReport | None
    attempts: int
    issues: list[str]
    llm_error: str | None
    best_spec: LabSpec | None
    best_report: ValidationReport | None
    # результат
    review: list[ReviewRemark]
    simulation: dict | None
    log: Annotated[list[str], operator.add]


def _llm(config: RunnableConfig) -> LabLLM | None:
    return (config.get("configurable") or {}).get("llm")


# --- узлы -------------------------------------------------------------------------


def plan(state: LabState, config: RunnableConfig) -> dict:
    llm = _llm(config)
    mode = state.get("mode", "auto")
    if mode == "auto":
        mode = "llm" if llm is not None and not state.get("topic_id") else "catalog"
    if mode == "llm" and llm is None:
        raise LLMError("LLM не подключена: добавьте ключ Claude/OpenAI или локальную модель в настройках аккаунта")
    update: dict[str, Any] = {"mode": mode, "attempts": 0, "issues": [], "llm_error": None}
    if mode == "llm":
        example = pick_example(state["request"], state["level"])
        update["example"] = example
        update["log"] = [f"Режим LLM ({llm.name}); образец из каталога — «{example.title}», уровень {state['level']}."]
    else:
        update["log"] = ["Режим каталога."]
    return update


def catalog(state: LabState) -> dict:
    spec = catalog_spec(state["request"], state["level"], state.get("topic_id"))
    return {"spec": spec, "log": [f"Каталог: выбрана тема «{spec.title}» (уровень {spec.level})."]}


def generate(state: LabState, config: RunnableConfig) -> dict:
    llm = _llm(config)
    level = get_level(state["level"])
    attempt = state.get("attempts", 0) + 1
    issues = state.get("issues") or []
    if issues and state.get("spec") is not None:
        spec_json = json.dumps(state["spec"].model_dump(exclude={"topic_id", "level"}), ensure_ascii=False, indent=1)
        messages = prompts.repair_messages(state["request"], level, spec_json, issues)
        action = "исправление по замечаниям"
    elif issues:
        messages = prompts.repair_messages(state["request"], level, (state.get("raw") or "")[:20000], issues)
        action = "повторная генерация"
    else:
        messages = prompts.generation_messages(state["request"], level, state["example"], state.get("materials"))
        action = "генерация"
    try:
        raw = llm.complete(messages, json=True)
    except LLMError as exc:
        return {"attempts": attempt, "llm_error": str(exc), "log": [f"Попытка {attempt} ({action}): ошибка модели — {exc}"]}
    try:
        spec = parse_spec(raw, state["level"])
    except Exception as exc:  # JSON или схема
        text = parse_error_text(exc)
        return {
            "attempts": attempt, "raw": raw, "spec": None, "llm_error": None,
            "issues": [f"Ответ не является корректной спецификацией: {text}"],
            "log": [f"Попытка {attempt} ({action}): ответ не разобран — {text}"],
        }
    return {"attempts": attempt, "raw": raw, "spec": spec, "llm_error": None,
            "log": [f"Попытка {attempt} ({action}): получена работа «{spec.title}»."]}


def validate(state: LabState) -> dict:
    report = validate_spec(state["spec"])
    update: dict[str, Any] = {"report": report, "issues": validation_issues(report), "log": [report_line(report)]}
    best = state.get("best_report")
    if best is None or quality(report) > quality(best):
        update["best_spec"] = state["spec"]
        update["best_report"] = report
    return update


def finalize(state: LabState) -> dict:
    if state.get("best_spec") is None:
        raise LLMError(state.get("llm_error") or "Модель не вернула ни одной корректной спецификации")
    return {
        "spec": state["best_spec"],
        "report": state["best_report"],
        "log": ["Попытки исчерпаны: сохранена лучшая версия как черновик с ошибками — исправьте её в редакторе."],
    }


def review(state: LabState, config: RunnableConfig) -> dict:
    log: list[str] = []
    llm = _llm(config) if state.get("mode") == "llm" else None
    remarks = review_spec(state["spec"], llm, log)
    log.append(f"Замечаний рецензента: {len(remarks)}.")
    return {"review": remarks, "log": log}


def simulate(state: LabState, config: RunnableConfig) -> dict:
    llm = _llm(config)
    n = state.get("simulate", 0)
    if n <= 0 or llm is None:
        return {"simulation": None, "log": []}
    result = simulate_students(llm, state["spec"], n=n)
    return {
        "simulation": result,
        "log": [f"Симуляция студентов: решили {result['solved']} из {result['n']}, "
                f"в среднем пройдено {result['mean_pass_rate']:.0%} тестов."],
    }


# --- переходы ---------------------------------------------------------------------


def route_after_plan(state: LabState) -> str:
    return "generate" if state["mode"] == "llm" else "catalog"


def route_after_generate(state: LabState) -> str:
    if state.get("spec") is not None and not state.get("llm_error"):
        return "validate"
    if state.get("llm_error") is None and state.get("attempts", 0) < state.get("max_attempts", 3):
        return "generate"
    return "finalize"


def route_after_validate(state: LabState) -> str:
    if state["report"].passed:
        return "review"
    if state.get("mode") == "llm" and state.get("attempts", 0) < state.get("max_attempts", 3):
        return "generate"
    return "finalize" if state.get("mode") == "llm" else "review"


def build_graph():
    graph = StateGraph(LabState)
    graph.add_node("plan", plan)
    graph.add_node("catalog", catalog)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("finalize", finalize)
    graph.add_node("review", review)
    graph.add_node("simulate", simulate)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, ["catalog", "generate"])
    graph.add_edge("catalog", "validate")
    graph.add_conditional_edges("generate", route_after_generate, ["validate", "generate", "finalize"])
    graph.add_conditional_edges("validate", route_after_validate, ["review", "generate", "finalize"])
    graph.add_edge("finalize", "review")
    graph.add_edge("review", "simulate")
    graph.add_edge("simulate", END)
    return graph.compile()


LAB_AGENT = build_graph()


# --- запуск -----------------------------------------------------------------------


@dataclass
class GenerationResult:
    spec: LabSpec
    report: ValidationReport
    mode: str
    review: list[ReviewRemark] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    simulation: dict | None = None
    llm_name: str | None = None


ProgressCallback = Callable[[str, list[str]], None]


def generate_lab(
    request: str,
    level: int,
    *,
    mode: Mode = "auto",
    llm: LabLLM | None = None,
    materials: str | None = None,
    topic_id: str | None = None,
    max_attempts: int = 3,
    simulate: int = 0,
    on_step: ProgressCallback | None = None,
) -> GenerationResult:
    """Запускает граф и возвращает результат. ``on_step(node, log_lines)`` — прогресс по шагам."""
    get_level(level)
    if mode not in ("auto", "catalog", "llm"):
        raise ValueError(f"Неизвестный режим генерации: {mode}")
    state: LabState = {
        "request": request,
        "level": level,
        "mode": mode,
        "topic_id": topic_id,
        "materials": materials,
        "max_attempts": max(1, max_attempts),
        "simulate": simulate,
        "log": [],
    }
    config: RunnableConfig = {"configurable": {"llm": llm}, "recursion_limit": 60}
    final: dict = {}
    for stream_mode, chunk in LAB_AGENT.stream(state, config, stream_mode=["updates", "values"]):
        if stream_mode == "values":
            final = chunk
        elif on_step is not None:
            for node, update in chunk.items():
                on_step(node, list((update or {}).get("log", [])))
    return GenerationResult(
        spec=final["spec"],
        report=final["report"],
        mode=final["mode"],
        review=final.get("review", []),
        log=final.get("log", []),
        simulation=final.get("simulation"),
        llm_name=llm.name if llm is not None and final["mode"] == "llm" else None,
    )


__all__ = ["LAB_AGENT", "NODE_LABELS", "GenerationResult", "TopicNotFound", "build_graph", "generate_lab"]

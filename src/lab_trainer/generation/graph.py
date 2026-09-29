"""Topic -> LabAssignment generation as a LangGraph state graph.

    START -> plan -> materials -> tests -> validate --ok--------------> END
                                              ^   \\--failed, budget--> repair
                                              \\___________________________/

    plan        topic + difficulty -> title, objectives, requirements (LabPlan)
    materials   plan -> brief, instructions, starter files, reference solution
    tests       plan + solution -> pytest files mapped to requirements, rubric
    validate    assemble the LabAssignment and run `validate_lab`: the solution must
                pass every test, the starter none, auto-check coverage >= 80%
    repair      errors + pytest output go back to the model, which returns corrected
                parts; loops back to validate at most `max_repairs` times

Each LLM node is `prompt | llm.with_structured_output(Schema)`, so the provider is
whatever LangChain chat model is passed in (see `llm.py`). If the repair budget runs
out, the draft is still returned with its errors so the teacher can fix it by hand.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from lab_trainer.generation import prompts
from lab_trainer.generation.validation import validate_lab
from lab_trainer.models import (
    AutoTest,
    Difficulty,
    LabAssignment,
    LabFile,
    Requirement,
    RubricItem,
)

MAX_REPAIR_ATTEMPTS = 3


class GenerationRequest(BaseModel):
    topic: str
    difficulty: Difficulty = Difficulty.BASIC
    language: str = "ru"
    teacher_notes: str = ""  # optional extra constraints ("use LangGraph", "no web access")


class LabPlan(BaseModel):
    """Output of the plan node."""

    id: str
    title: str
    learning_objectives: list[str]
    requirements: list[Requirement]
    estimated_minutes: int


class LabMaterials(BaseModel):
    """Output of the materials node."""

    brief: str
    instructions: str
    starter_files: list[LabFile]
    solution_files: list[LabFile]


class LabSuite(BaseModel):
    """Output of the tests node."""

    test_files: list[LabFile]
    tests: list[AutoTest]
    rubric: list[RubricItem]


class RepairPatch(BaseModel):
    """Output of the repair node: full replacements for the parts that needed fixing."""

    plan: LabPlan | None = None
    materials: LabMaterials | None = None
    suite: LabSuite | None = None


class GenerationState(TypedDict, total=False):
    request: GenerationRequest
    plan: LabPlan
    materials: LabMaterials
    suite: LabSuite
    lab: LabAssignment | None  # None while the parts do not assemble into a valid model
    errors: list[str]
    test_output: str
    repairs: int


class GenerationResult(BaseModel):
    lab: LabAssignment | None
    errors: list[str]  # empty when the lab passed validation
    repairs: int


def assemble(request: GenerationRequest, plan: LabPlan, m: LabMaterials, s: LabSuite) -> LabAssignment:
    """Merge the node outputs; requirement -> test links are derived from the test index."""
    requirements = [
        r.model_copy(update={"test_ids": [t.id for t in s.tests if r.id in t.requirement_ids]})
        for r in plan.requirements
    ]
    return LabAssignment(
        id=plan.id,
        title=plan.title,
        topic=request.topic,
        difficulty=request.difficulty,
        language=request.language,
        estimated_minutes=plan.estimated_minutes,
        learning_objectives=plan.learning_objectives,
        requirements=requirements,
        brief=m.brief,
        instructions=m.instructions,
        starter_files=m.starter_files,
        solution_files=m.solution_files,
        test_files=s.test_files,
        tests=s.tests,
        rubric=s.rubric,
    )


def _dump(model: BaseModel | list[LabFile]) -> str:
    if isinstance(model, list):
        return json.dumps([f.model_dump() for f in model], ensure_ascii=False, indent=2)
    return model.model_dump_json(indent=2)


def build_graph(llm: BaseChatModel, max_repairs: int = MAX_REPAIR_ATTEMPTS, run_tests: bool = True):
    def structured(prompt, schema: type[BaseModel]):
        # json_schema = native structured outputs (Anthropic, OpenAI, Ollama); forced tool
        # calling is not supported by current Claude models
        return prompt | llm.with_structured_output(schema, method="json_schema")

    def common(state: GenerationState) -> dict:
        request = state["request"]
        return {
            "difficulty": request.difficulty.value,
            "difficulty_guide": prompts.DIFFICULTY_GUIDE[request.difficulty.value],
        }

    def plan(state: GenerationState) -> dict:
        request = state["request"]
        result = structured(prompts.PLAN, LabPlan).invoke({
            **common(state),
            "topic": request.topic,
            "language": request.language,
            "teacher_notes": request.teacher_notes or "нет",
        })
        return {"plan": result}

    def materials(state: GenerationState) -> dict:
        result = structured(prompts.MATERIALS, LabMaterials).invoke(
            {**common(state), "plan": _dump(state["plan"])}
        )
        return {"materials": result}

    def tests(state: GenerationState) -> dict:
        m = state["materials"]
        result = structured(prompts.TESTS, LabSuite).invoke({
            **common(state),
            "plan": _dump(state["plan"]),
            "solution": _dump(m.solution_files),
            "starter": _dump(m.starter_files),
        })
        return {"suite": result}

    def validate(state: GenerationState) -> dict:
        try:
            lab = assemble(state["request"], state["plan"], state["materials"], state["suite"])
        except ValidationError as exc:
            return {"lab": None, "errors": [str(exc)], "test_output": ""}
        report = validate_lab(lab, run_tests=run_tests)
        return {"lab": lab, "errors": report.errors, "test_output": report.output}

    def repair(state: GenerationState) -> dict:
        patch = structured(prompts.REPAIR, RepairPatch).invoke({
            "errors": "\n".join(f"- {e}" for e in state["errors"]),
            "test_output": state.get("test_output") or "нет",
            "plan": _dump(state["plan"]),
            "materials": _dump(state["materials"]),
            "suite": _dump(state["suite"]),
        })
        update: dict = {"repairs": state.get("repairs", 0) + 1}
        for part in ("plan", "materials", "suite"):
            if (value := getattr(patch, part)) is not None:
                update[part] = value
        return update

    def after_validate(state: GenerationState) -> str:
        if not state["errors"] or state.get("repairs", 0) >= max_repairs:
            return END
        return "repair"

    graph = StateGraph(GenerationState)
    graph.add_node("plan", plan)
    graph.add_node("materials", materials)
    graph.add_node("tests", tests)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "materials")
    graph.add_edge("materials", "tests")
    graph.add_edge("tests", "validate")
    graph.add_conditional_edges("validate", after_validate, ["repair", END])
    graph.add_edge("repair", "validate")
    return graph.compile()


def generate_lab(
    request: GenerationRequest, llm: BaseChatModel, max_repairs: int = MAX_REPAIR_ATTEMPTS
) -> GenerationResult:
    state = build_graph(llm, max_repairs).invoke({"request": request, "repairs": 0})
    return GenerationResult(lab=state["lab"], errors=state["errors"], repairs=state["repairs"])

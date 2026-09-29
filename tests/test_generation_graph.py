"""The LangGraph generation graph, driven by a scripted chat model (no API key needed)."""

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from lab_trainer.generation import GenerationRequest, generate_lab
from lab_trainer.generation.graph import LabMaterials, LabPlan, LabSuite, RepairPatch
from lab_trainer.models import LabFile
from lab_trainer.storage import load_lab

EXAMPLE = Path(__file__).parents[1] / "labs" / "tool-agent-basic"


class ScriptedChatModel:
    """Stands in for a LangChain chat model: `with_structured_output(Schema)` returns the
    next scripted object for that schema and records the rendered prompt."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.prompts: list[tuple[str, str]] = []

    def with_structured_output(self, schema, **kwargs):
        def respond(prompt_value):
            self.prompts.append((schema.__name__, prompt_value.to_string()))
            answer = self.script.pop(0)
            assert isinstance(answer, schema), f"expected {schema.__name__}, got {answer!r}"
            return answer

        return RunnableLambda(respond)


@pytest.fixture
def example():
    lab = load_lab(EXAMPLE)
    plan = LabPlan(
        id=lab.id,
        title=lab.title,
        learning_objectives=lab.learning_objectives,
        requirements=[r.model_copy(update={"test_ids": []}) for r in lab.requirements],
        estimated_minutes=lab.estimated_minutes,
    )
    materials = LabMaterials(
        brief=lab.brief,
        instructions=lab.instructions,
        starter_files=lab.starter_files,
        solution_files=lab.solution_files,
    )
    suite = LabSuite(test_files=lab.test_files, tests=lab.tests, rubric=lab.rubric)
    request = GenerationRequest(topic=lab.topic, difficulty=lab.difficulty)
    return lab, request, plan, materials, suite


def broken(materials: LabMaterials) -> LabMaterials:
    """Reference solution whose calculator is wrong."""
    solution = [
        LabFile(path=f.path, content=f.content.replace("operator.add", "operator.sub"))
        for f in materials.solution_files
    ]
    return materials.model_copy(update={"solution_files": solution})


def test_valid_lab_needs_no_repair(example):
    lab, request, plan, materials, suite = example
    llm = ScriptedChatModel([plan, materials, suite])
    result = generate_lab(request, llm)
    assert result.errors == [] and result.repairs == 0
    assert result.lab == lab
    assert [name for name, _ in llm.prompts] == ["LabPlan", "LabMaterials", "LabSuite"]


def test_failed_validation_is_repaired(example):
    lab, request, plan, materials, suite = example
    llm = ScriptedChatModel([plan, broken(materials), suite, RepairPatch(materials=materials)])
    result = generate_lab(request, llm)
    assert result.errors == [] and result.repairs == 1
    assert result.lab == lab
    name, repair_prompt = llm.prompts[-1]
    assert name == "RepairPatch" and "reference solution fails tests for R1" in repair_prompt


def test_repair_budget_returns_draft_with_errors(example):
    _, request, plan, materials, suite = example
    llm = ScriptedChatModel([plan, broken(materials), suite, RepairPatch()])
    result = generate_lab(request, llm, max_repairs=1)
    assert result.repairs == 1
    assert result.lab is not None
    assert result.errors == ["reference solution fails tests for R1"]


def test_inconsistent_parts_go_to_repair(example):
    lab, request, plan, materials, suite = example
    bad_suite = suite.model_copy(update={"tests": [
        t.model_copy(update={"file": "missing.py"}) for t in suite.tests
    ]})
    llm = ScriptedChatModel([plan, materials, bad_suite, RepairPatch(suite=suite)])
    result = generate_lab(request, llm)
    assert result.errors == [] and result.repairs == 1
    assert result.lab == lab

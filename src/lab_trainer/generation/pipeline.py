"""Topic -> LabAssignment generation pipeline.

Stages (each a separate LLM call with a structured-output schema, so every stage
can be retried or regenerated on its own after teacher feedback):

    1. plan        topic + difficulty -> title, objectives, requirements (LabPlan)
    2. materials   plan -> brief, instructions, starter files, reference solution
    3. tests       plan + solution -> pytest files mapped to requirements, rubric
    4. validate    run tests in the sandbox: solution must pass, starter must fail
                   the auto-checked tests; coverage of auto checks must be >= 80%
    5. repair      on validation failure, feed the pytest output back to stage 2/3
                   (bounded number of attempts)

Stages 1-3 are stubbed here; stage 4 lives in `validation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from lab_trainer.generation.llm import LLMClient
from lab_trainer.models import Difficulty, LabAssignment, Requirement

MAX_REPAIR_ATTEMPTS = 3


class LabPlan(BaseModel):
    """Output of stage 1."""

    id: str
    title: str
    learning_objectives: list[str]
    requirements: list[Requirement]
    estimated_minutes: int


@dataclass
class GenerationRequest:
    topic: str
    difficulty: Difficulty
    language: str = "ru"
    teacher_notes: str = ""  # optional extra constraints ("use LangGraph", "no web access")


class GenerationPipeline:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, request: GenerationRequest) -> LabAssignment:
        plan = self.plan(request)
        lab = self.materials(request, plan)
        lab = self.tests(lab)
        # TODO: validation.validate_lab(lab) + repair loop up to MAX_REPAIR_ATTEMPTS
        return lab

    def plan(self, request: GenerationRequest) -> LabPlan:
        raise NotImplementedError("stage 1: prompts.PLAN_PROMPT -> self.llm.structured(LabPlan)")

    def materials(self, request: GenerationRequest, plan: LabPlan) -> LabAssignment:
        raise NotImplementedError("stage 2: brief, instructions, starter and solution files")

    def tests(self, lab: LabAssignment) -> LabAssignment:
        raise NotImplementedError("stage 3: pytest files + AutoTest index + rubric")

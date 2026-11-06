"""Оркестратор: тема преподавателя → готовая лабораторная работа с отчётами.

Схема конвейера:

1. генератор (каталог или LLM) создаёт задание, шаблон, эталон, тесты и рубрику;
2. валидатор запускает эталон, шаблон и мутанты в песочнице; при ошибках LLM-генератор
   получает замечания и исправляет работу;
3. рецензент готовит замечания для преподавателя (без баллов);
4. по желанию — симулированные студенты оценивают понятность и сложность;
5. преподаватель редактирует и утверждает работу перед публикацией.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..llm.base import LLMProvider
from ..models import LabSpec, ReviewRemark, ValidationReport
from .generator import CatalogGenerator, LLMGenerator, TopicNotFound
from .reviewer import review_spec
from .students import simulate_students

Mode = Literal["auto", "catalog", "llm"]


@dataclass
class GenerationResult:
    spec: LabSpec
    report: ValidationReport
    mode: str
    review: list[ReviewRemark] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    simulation: dict | None = None


def generate_lab(
    request: str,
    level: int,
    *,
    mode: Mode = "auto",
    provider: LLMProvider | None = None,
    materials: str | None = None,
    topic_id: str | None = None,
    max_attempts: int = 3,
    simulate: int = 0,
) -> GenerationResult:
    if mode == "auto":
        mode = "llm" if provider is not None and topic_id is None else "catalog"
    if mode == "llm":
        if provider is None:
            raise ValueError("LLM не настроена: задайте LABGEN_LLM_BASE_URL и LABGEN_LLM_MODEL или выберите режим каталога")
        output = LLMGenerator(provider, max_attempts=max_attempts).generate(request, level, materials)
    elif mode == "catalog":
        output = CatalogGenerator().generate(request, level, topic_id=topic_id)
    else:
        raise ValueError(f"Неизвестный режим генерации: {mode}")

    log = list(output.log)
    review = review_spec(output.spec, provider if mode == "llm" else None, log)
    simulation = None
    if simulate > 0 and provider is not None:
        simulation = simulate_students(provider, output.spec, n=simulate)
        log.append(
            f"Симуляция студентов: решили {simulation['solved']} из {simulation['n']}, "
            f"средняя доля пройденных тестов {simulation['mean_pass_rate']:.0%}."
        )
    return GenerationResult(
        spec=output.spec, report=output.report, mode=mode, review=review, log=log, simulation=simulation
    )


__all__ = ["GenerationResult", "TopicNotFound", "generate_lab"]

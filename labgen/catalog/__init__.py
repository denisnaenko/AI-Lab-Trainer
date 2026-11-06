"""Каталог проверенных лабораторных работ по агентной разработке (5 тем × 3 уровня).

Каталог используется:

* в офлайн-режиме — как источник готовых заданий по теме преподавателя;
* в LLM-режиме — как образцы (few-shot) для генерации новых заданий.
"""

from __future__ import annotations

import re

from ..models import LabSpec
from .base import TopicDef
from .memory_agent import TOPIC as MEMORY_AGENT
from .multi_agent import TOPIC as MULTI_AGENT
from .rag_agent import TOPIC as RAG_AGENT
from .structured_output import TOPIC as STRUCTURED_OUTPUT
from .tool_agent import TOPIC as TOOL_AGENT

TOPICS: dict[str, TopicDef] = {
    t.id: t for t in (STRUCTURED_OUTPUT, TOOL_AGENT, MEMORY_AGENT, RAG_AGENT, MULTI_AGENT)
}


def list_topics() -> list[TopicDef]:
    return list(TOPICS.values())


def get_topic(topic_id: str) -> TopicDef:
    if topic_id not in TOPICS:
        raise KeyError(f"Нет темы {topic_id!r}. Доступные: {', '.join(TOPICS)}")
    return TOPICS[topic_id]


def _normalize(text: str) -> str:
    return " " + " ".join(re.findall(r"[\w-]+", text.lower().replace("ё", "е"))) + " "


def score_topics(request: str) -> list[tuple[TopicDef, float]]:
    """Оценка соответствия темы запроса каждой теме каталога (по ключевым словам)."""
    text = _normalize(request)
    results = []
    for topic in TOPICS.values():
        score = 0.0
        for keyword in topic.keywords + (topic.title,):
            key = _normalize(keyword).strip()
            if not key:
                continue
            if f" {key} " in text:
                score += 1.0 + 0.5 * key.count(" ")
            elif len(key) >= 5 and key in text:
                score += 0.5  # совпадение по основе слова: «инструмент» в «инструментами»
        results.append((topic, score))
    return sorted(results, key=lambda item: -item[1])


MIN_MATCH_SCORE = 1.0


def match_topic(request: str) -> TopicDef | None:
    """Лучшая тема каталога; нужна хотя бы одна точная фраза из ключевых слов."""
    best, score = score_topics(request)[0]
    return best if score >= MIN_MATCH_SCORE else None


def build_lab(topic_id: str, level: int) -> LabSpec:
    return get_topic(topic_id).build(level)

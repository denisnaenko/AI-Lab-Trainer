"""ИИ-агент генерации лабораторных работ на LangGraph."""

from .graph import LAB_AGENT, NODE_LABELS, GenerationResult, build_graph, generate_lab

__all__ = ["LAB_AGENT", "NODE_LABELS", "GenerationResult", "build_graph", "generate_lab"]

"""Лабораторная работа «RAG: ответы по базе знаний», уровень 2.

Документ — словарь ``{"id": "d1", "title": "...", "text": "..."}``.
"""

import re

STOPWORDS = frozenset(
    "и в во на с со по к о об от до за из у не что как это для ли а но или же то ну "
    "мне я ты вы мы он она они мой ваш какой какая какие где когда есть".split()
)
RAG_SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы студентов только по приведённому контексту. "
    "Ссылайся на документы в квадратных скобках, например [doc1]. "
    "Если в контексте нет ответа, так и скажи."
)
NO_ANSWER = "В базе знаний нет информации по этому вопросу."


def tokenize(text):
    """Слова в нижнем регистре без стоп-слов."""
    raise NotImplementedError


class KnowledgeBase:
    def __init__(self, documents=()):
        raise NotImplementedError

    def add(self, document):
        """Добавить документ; id должен быть уникальным."""
        raise NotImplementedError

    def search(self, query, top_k=3):
        raise NotImplementedError


class RAGAgent:
    def __init__(self, llm, knowledge_base, top_k=3):
        self.llm = llm
        self.kb = knowledge_base
        self.top_k = top_k

    def answer(self, question):
        """Если подходящих документов нет — не тратить вызов модели."""
        raise NotImplementedError

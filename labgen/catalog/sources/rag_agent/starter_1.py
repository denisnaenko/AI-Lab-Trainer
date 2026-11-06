"""Лабораторная работа «RAG: ответы по базе знаний», уровень 1.

Документ — словарь ``{"id": "d1", "title": "...", "text": "..."}``.
Модель: ``response = llm.chat(messages)``, текст ответа — ``response.content``.
"""

import re

RAG_SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы студентов только по приведённому контексту. "
    "Ссылайся на документы в квадратных скобках, например [doc1]. "
    "Если в контексте нет ответа, так и скажи."
)


def tokenize(text):
    """Разбить текст на слова в нижнем регистре."""
    # TODO: используйте re.findall(r"\w+", ...) и str.lower()
    raise NotImplementedError


class KnowledgeBase:
    def __init__(self, documents):
        self.documents = list(documents)

    def search(self, query, top_k=3):
        """Вернуть до top_k документов с наибольшим числом общих слов с запросом."""
        # TODO: 1. Получите множество слов запроса.
        # TODO: 2. Для каждого документа посчитайте число общих слов с заголовком и текстом.
        # TODO: 3. Отбросьте документы без совпадений.
        # TODO: 4. Отсортируйте по убыванию совпадений (при равенстве — в исходном порядке)
        #          и верните первые top_k документов.
        raise NotImplementedError


class RAGAgent:
    def __init__(self, llm, knowledge_base, top_k=3):
        self.llm = llm
        self.kb = knowledge_base
        self.top_k = top_k

    def answer(self, question):
        """Вернуть {"answer": текст ответа модели, "sources": [id найденных документов]}."""
        # TODO: 1. Найдите документы: self.kb.search(question, self.top_k).
        # TODO: 2. Соберите контекст: по строке на документ вида "[d1] Заголовок: текст".
        # TODO: 3. Отправьте модели системное сообщение RAG_SYSTEM_PROMPT и сообщение пользователя
        #          с контекстом и вопросом.
        # TODO: 4. Верните ответ модели и id найденных документов.
        raise NotImplementedError

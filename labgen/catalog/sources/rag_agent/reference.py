"""Эталонное решение: агент, отвечающий на вопросы по базе знаний (RAG)."""

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
REPHRASE_PROMPT = (
    "Переформулируй вопрос другими словами, используя синонимы и ключевые термины предметной области. "
    "Верни только новый вопрос."
)


def tokenize(text):
    """Слова текста в нижнем регистре без стоп-слов."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class KnowledgeBase:
    """Простейший поиск по совпадению слов."""

    def __init__(self, documents=()):
        self.documents = []
        self._tokens = []
        for document in documents:
            self.add(document)

    def add(self, document):
        if any(d["id"] == document["id"] for d in self.documents):
            raise ValueError(f"документ с id {document['id']!r} уже есть в базе")
        self.documents.append(document)
        self._tokens.append(set(tokenize(document["title"] + " " + document["text"])))

    def search(self, query, top_k=3):
        """До top_k документов с наибольшим числом общих слов с запросом."""
        query_tokens = set(tokenize(query))
        scored = []
        for index, (document, tokens) in enumerate(zip(self.documents, self._tokens)):
            score = len(query_tokens & tokens)
            if score > 0:
                scored.append((score, index, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[:top_k]]


class RAGAgent:
    """Находит документы, передаёт их модели как контекст и возвращает ответ с источниками."""

    def __init__(self, llm, knowledge_base, top_k=3, rephrase=False):
        self.llm = llm
        self.kb = knowledge_base
        self.top_k = top_k
        self.rephrase = rephrase

    def build_prompt(self, question, documents):
        context = "\n".join(f"[{d['id']}] {d['title']}: {d['text']}" for d in documents)
        return [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]

    def _retrieve(self, question):
        documents = self.kb.search(question, self.top_k)
        if documents or not self.rephrase:
            return documents
        response = self.llm.chat(
            [
                {"role": "system", "content": REPHRASE_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        new_question = (response.content or "").strip()
        return self.kb.search(new_question, self.top_k) if new_question else []

    def _cited_sources(self, answer, documents):
        found = [d["id"] for d in documents]
        cited = [doc_id for doc_id in found if f"[{doc_id}]" in answer]
        return cited or found

    def answer(self, question):
        documents = self._retrieve(question)
        if not documents:
            return {"answer": NO_ANSWER, "sources": []}
        response = self.llm.chat(self.build_prompt(question, documents))
        answer = response.content or ""
        return {"answer": answer, "sources": self._cited_sources(answer, documents)}

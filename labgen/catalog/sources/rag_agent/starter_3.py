"""Лабораторная работа «RAG: ответы по базе знаний», уровень 3.

Реализуйте публичный интерфейс ниже; внутреннюю структуру спроектируйте сами.
"""

STOPWORDS = frozenset(
    "и в во на с со по к о об от до за из у не что как это для ли а но или же то ну "
    "мне я ты вы мы он она они мой ваш какой какая какие где когда есть".split()
)
RAG_SYSTEM_PROMPT = "Отвечай только по контексту и ссылайся на документы в квадратных скобках, например [doc1]."
NO_ANSWER = "В базе знаний нет информации по этому вопросу."
REPHRASE_PROMPT = "Переформулируй вопрос другими словами. Верни только новый вопрос."


def tokenize(text):
    raise NotImplementedError


class KnowledgeBase:
    def __init__(self, documents=()):
        raise NotImplementedError

    def add(self, document):
        raise NotImplementedError

    def search(self, query, top_k=3):
        raise NotImplementedError


class RAGAgent:
    def __init__(self, llm, knowledge_base, top_k=3, rephrase=False):
        raise NotImplementedError

    def answer(self, question):
        raise NotImplementedError

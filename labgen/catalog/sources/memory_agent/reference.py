"""Эталонное решение: чат-агент с памятью диалога, окном и сжатием истории."""

DEFAULT_SYSTEM_PROMPT = "Ты — дружелюбный ассистент. Учитывай весь предыдущий диалог."
SUMMARY_PROMPT = (
    "Кратко перескажи диалог ниже: сохрани факты о пользователе, договорённости и открытые вопросы. "
    "Пиши от третьего лица, не больше пяти предложений."
)
SUMMARY_PREFIX = "Краткое содержание предыдущего диалога: "
KEEP_LAST = 2


class ChatAgent:
    """Агент, который помнит диалог и умеет его сжимать."""

    def __init__(self, llm, system_prompt=DEFAULT_SYSTEM_PROMPT, max_history=None, summarize_threshold=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.summarize_threshold = summarize_threshold
        self.history = []
        self.summary = None

    def reset(self):
        """Начать диалог заново."""
        self.history = []
        self.summary = None

    def _context(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": SUMMARY_PREFIX + self.summary})
        return messages + self.history

    def _trim(self):
        if self.max_history is None or len(self.history) <= self.max_history:
            return
        trimmed = self.history[-self.max_history:]
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]
        self.history = trimmed

    def _transcript(self, messages):
        lines = []
        if self.summary:
            lines.append(f"Ранее: {self.summary}")
        names = {"user": "Пользователь", "assistant": "Ассистент"}
        lines += [f"{names.get(m['role'], m['role'])}: {m['content']}" for m in messages]
        return "\n".join(lines)

    def _summarize(self):
        old, recent = self.history[:-KEEP_LAST], self.history[-KEEP_LAST:]
        response = self.llm.chat(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": self._transcript(old)},
            ]
        )
        self.summary = (response.content or "").strip()
        self.history = recent

    def send(self, user_message):
        """Отправить сообщение пользователя и вернуть ответ модели."""
        if self.summarize_threshold is not None and len(self.history) > self.summarize_threshold:
            self._summarize()
        user = {"role": "user", "content": user_message}
        response = self.llm.chat(self._context() + [user])
        reply = response.content or ""
        self.history = self.history + [user, {"role": "assistant", "content": reply}]
        self._trim()
        return reply

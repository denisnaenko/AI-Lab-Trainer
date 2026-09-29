## Задание

1. Установите зависимости: `pip install -r requirements.txt`.
2. Откройте `agent.py`. Реализуйте инструмент `calculator(expression)` (он уже обёрнут в
   `@tool`): он вычисляет арифметическое выражение с `+ - * /` и скобками. Не используйте
   `eval` напрямую.
3. Реализуйте `run_agent(llm, user_message, max_steps=5)`, где `llm` — чат-модель LangChain:
   - привяжите инструменты: `model = llm.bind_tools(TOOLS)` и вызывайте `model.invoke(messages)`;
   - если в ответе (`AIMessage`) нет `tool_calls` — верните `response.content`;
   - иначе добавьте ответ модели в `messages`, выполните каждый вызов из `response.tool_calls`
     и добавьте `ToolMessage(str(результат), name=..., tool_call_id=call["id"])`;
   - неизвестный инструмент или исключение — тоже `ToolMessage` с текстом ошибки;
   - после `max_steps` обращений к модели верните строку, начинающуюся с `"Stopped"`.
4. Запустите тесты: `python -m pytest tests`.
5. В `REPORT.md` объясните, как ваш агент решает, когда остановиться.

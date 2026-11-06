import solution as sol
from agentkit import LLMResponse, ScriptedLLM, ToolCall, call, text, tool_call, tool_calls


def schema(name, **params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Инструмент {name}",
            "parameters": {
                "type": "object",
                "properties": {p: {"type": t} for p, t in params.items()},
                "required": list(params),
            },
        },
    }


SCHEMAS = [schema("get_weather", city="string"), schema("add", a="number", b="number")]


def make_agent(responses, **kwargs):
    """Агент с тестовыми инструментами; возвращает (агент, модель, журнал вызовов)."""
    log = []

    def get_weather(city):
        log.append(("get_weather", {"city": city}))
        return f"{city}: +3°C"

    def add(a, b):
        log.append(("add", {"a": a, "b": b}))
        return a + b

    def fail(**kwargs):
        raise ValueError("сервис недоступен")

    llm = ScriptedLLM(responses)
    tools = {"get_weather": get_weather, "add": add, "fail": fail}
    return sol.ToolAgent(llm, tools, SCHEMAS, **kwargs), llm, log


def tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


def test_r1_returns_text_answer():
    agent, llm, log = make_agent([text("Привет! Чем помочь?")])
    assert agent.run("Привет") == "Привет! Чем помочь?", "run должен вернуть текст ответа модели"
    assert llm.call_count == 1 and log == [], "без запроса инструментов нужен один вызов модели"


def test_r1_sends_system_user_and_tools():
    agent, llm, _ = make_agent([text("ок")], system_prompt="Особая инструкция")
    agent.run("Привет")
    messages = llm.messages_of(0)
    assert messages[0] == {"role": "system", "content": "Особая инструкция"}
    assert messages[-1] == {"role": "user", "content": "Привет"}
    assert llm.calls[0]["tools"] == SCHEMAS, "модели нужно передать схемы инструментов"


def test_r2_calls_tool_with_arguments():
    agent, _, log = make_agent([tool_call("get_weather", city="Казань"), text("В Казани +3°C")])
    agent.run("Какая погода в Казани?")
    assert log == [("get_weather", {"city": "Казань"})], "агент должен вызвать инструмент с аргументами из запроса модели"


def test_r2_numeric_arguments():
    agent, _, log = make_agent([tool_call("add", a=2, b=3), text("5")])
    agent.run("Сколько будет 2 + 3?")
    assert log == [("add", {"a": 2, "b": 3})], "аргументы нужно передать в функцию без изменений"


def test_r3_tool_result_sent_back():
    request = tool_call("get_weather", city="Казань")
    call_id = request.tool_calls[0].id
    agent, llm, _ = make_agent([request, text("В Казани +3°C")])
    assert agent.run("Какая погода в Казани?") == "В Казани +3°C", "нужно вернуть итоговый ответ модели"
    assert llm.call_count == 2, "после вызова инструмента модель нужно вызвать ещё раз"
    second = llm.messages_of(1)
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["name"] == "get_weather"
    assert second[-1]["role"] == "tool", "результат инструмента передаётся сообщением role=\"tool\""
    assert second[-1]["tool_call_id"] == call_id, "tool_call_id должен совпадать с id запроса"
    assert second[-1]["content"] == "Казань: +3°C", "в сообщении нужен результат инструмента"


def test_r3_history_preserved():
    agent, llm, _ = make_agent([tool_call("add", a=1, b=2), text("3")])
    agent.run("1 + 2?")
    assert llm.messages_of(1)[:2] == llm.messages_of(0), "история диалога должна сохраняться между вызовами"


def test_r4_parallel_tool_calls():
    request = tool_calls(call("get_weather", city="Москва"), call("add", a=1, b=2))
    ids = [c.id for c in request.tool_calls]
    agent, llm, log = make_agent([request, text("готово")])
    assert agent.run("Погода в Москве и 1 + 2") == "готово"
    assert log == [("get_weather", {"city": "Москва"}), ("add", {"a": 1, "b": 2})], (
        "нужно выполнить все вызовы инструментов из одного ответа модели по порядку"
    )
    results = tool_messages(llm.messages_of(1))
    assert [m["tool_call_id"] for m in results] == ids, "на каждый вызов нужен свой ответ role=\"tool\""
    assert [m["content"] for m in results] == ["Москва: +3°C", "3"]


def test_r4_multiple_rounds():
    agent, llm, log = make_agent(
        [tool_call("get_weather", city="Москва"), tool_call("add", a=1, b=2), text("готово")]
    )
    assert agent.run("задача в два шага") == "готово", "агент должен продолжать, пока модель запрашивает инструменты"
    assert llm.call_count == 3 and len(log) == 2


def test_r5_step_limit():
    agent, llm, _ = make_agent([tool_call("add", a=1, b=1)] * 3, max_steps=3)
    assert agent.run("зациклись") == sol.STEP_LIMIT_MESSAGE
    assert llm.call_count == 3, "агент не должен вызывать модель больше max_steps раз"


def test_r5_exact_limit_is_enough():
    agent, llm, _ = make_agent([tool_call("add", a=1, b=1), text("2")], max_steps=2)
    assert agent.run("1 + 1") == "2", "max_steps — число вызовов модели; двух вызовов здесь достаточно"


def test_r6_unknown_tool():
    agent, llm, _ = make_agent([tool_call("translate", text="hi"), text("Не могу перевести")])
    assert agent.run("Переведи hi") == "Не могу перевести"
    result = tool_messages(llm.messages_of(1))[0]["content"]
    assert "Неизвестный инструмент" in result and "translate" in result, "модели нужно сообщить о неизвестном инструменте"


def test_r6_unknown_then_known():
    request = tool_calls(call("translate", text="hi"), call("get_weather", city="Омск"))
    agent, llm, log = make_agent([request, text("ок")])
    agent.run("Переведи и скажи погоду")
    assert log == [("get_weather", {"city": "Омск"})], "после неизвестного инструмента остальные вызовы тоже нужно выполнить"
    assert len(tool_messages(llm.messages_of(1))) == 2, "на каждый запрос нужен ответ role=tool"


def test_r7_tool_exception_reported():
    agent, llm, _ = make_agent([tool_call("fail", x=1), text("Сервис недоступен, попробуйте позже")])
    assert agent.run("Сделай что-нибудь") == "Сервис недоступен, попробуйте позже"
    result = tool_messages(llm.messages_of(1))[0]["content"]
    assert result.startswith("Ошибка") and "сервис недоступен" in result, "модель должна получить текст «Ошибка: <описание>»"


def test_r7_json_string_arguments():
    request = LLMResponse(tool_calls=[ToolCall("add", '{"a": 4, "b": 5}')])
    agent, llm, log = make_agent([request, text("9")])
    assert agent.run("4 + 5") == "9"
    assert log == [("add", {"a": 4, "b": 5})], "аргументы-строку JSON нужно разобрать в словарь"
    assert tool_messages(llm.messages_of(1))[0]["content"] == "9"


def test_r7_invalid_json_arguments():
    request = LLMResponse(tool_calls=[ToolCall("add", "{a: 1")])
    agent, llm, log = make_agent([request, text("Не получилось")])
    assert agent.run("сломай аргументы") == "Не получилось"
    assert log == []
    assert tool_messages(llm.messages_of(1))[0]["content"].startswith("Ошибка")


def test_r8_trace_records_calls():
    agent, _, _ = make_agent([tool_call("get_weather", city="Казань"), text("ок")])
    agent.run("Погода в Казани")
    assert agent.trace == [{"tool": "get_weather", "arguments": {"city": "Казань"}, "result": "Казань: +3°C"}], (
        "в trace нужна запись {tool, arguments, result} на каждый вызов"
    )


def test_r8_trace_reset_between_runs():
    agent, _, _ = make_agent([tool_call("add", a=1, b=2), text("3"), text("без инструментов")])
    agent.run("1 + 2")
    agent.run("просто вопрос")
    assert agent.trace == [], "трасса должна очищаться при каждом запуске"


def test_r8_trace_includes_errors():
    agent, _, _ = make_agent([tool_call("translate", text="hi"), text("ок")])
    agent.run("Переведи")
    assert len(agent.trace) == 1
    assert agent.trace[0]["tool"] == "translate" and "Неизвестный инструмент" in agent.trace[0]["result"]

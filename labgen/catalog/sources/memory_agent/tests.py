import solution as sol
from agentkit import ScriptedLLM, text

SYSTEM = "Ты — тестовый бот."


def make_agent(responses, **kwargs):
    llm = ScriptedLLM(responses)
    kwargs.setdefault("system_prompt", SYSTEM)
    return sol.ChatAgent(llm, **kwargs), llm


def user(content):
    return {"role": "user", "content": content}


def bot(content):
    return {"role": "assistant", "content": content}


def system(content=SYSTEM):
    return {"role": "system", "content": content}


def raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"ожидалось исключение {exc_type.__name__}")


def test_r1_first_message():
    agent, llm = make_agent([text("Здравствуйте!")])
    assert agent.send("Привет") == "Здравствуйте!"
    assert llm.messages_of(0) == [system(), user("Привет")]


def test_r1_default_system_prompt():
    llm = ScriptedLLM([text("ок")])
    agent = sol.ChatAgent(llm)
    agent.send("Привет")
    assert llm.messages_of(0)[0] == {"role": "system", "content": sol.DEFAULT_SYSTEM_PROMPT}


def test_r2_history_sent_on_next_call():
    agent, llm = make_agent([text("Приятно познакомиться, Аня!"), text("Вас зовут Аня.")])
    agent.send("Меня зовут Аня")
    assert agent.send("Как меня зовут?") == "Вас зовут Аня."
    assert llm.messages_of(1) == [
        system(),
        user("Меня зовут Аня"),
        bot("Приятно познакомиться, Аня!"),
        user("Как меня зовут?"),
    ]


def test_r2_history_attribute():
    agent, _ = make_agent([text("а1"), text("а2")])
    agent.send("в1")
    agent.send("в2")
    assert agent.history == [user("в1"), bot("а1"), user("в2"), bot("а2")]


def test_r3_reset():
    agent, llm = make_agent([text("а1"), text("а2")])
    agent.send("в1")
    agent.reset()
    agent.send("в2")
    assert llm.messages_of(1) == [system(), user("в2")]
    assert agent.history == [user("в2"), bot("а2")]


def test_r4_window():
    agent, llm = make_agent([text("а1"), text("а2"), text("а3")], max_history=2)
    agent.send("в1")
    agent.send("в2")
    agent.send("в3")
    assert llm.messages_of(2) == [system(), user("в2"), bot("а2"), user("в3")]
    assert agent.history == [user("в3"), bot("а3")]


def test_r4_no_limit_by_default():
    agent, llm = make_agent([text(f"а{i}") for i in range(4)])
    for i in range(4):
        agent.send(f"в{i}")
    assert len(llm.messages_of(3)) == 8, "без max_history модель должна видеть весь диалог"


def test_r5_window_starts_with_user():
    agent, _ = make_agent([text("а1"), text("а2")], max_history=3)
    agent.send("в1")
    agent.send("в2")
    assert agent.history == [user("в2"), bot("а2")], "история не должна начинаться с ответа ассистента"


def test_r5_odd_window_next_call():
    agent, llm = make_agent([text("а1"), text("а2"), text("а3")], max_history=3)
    for message in ("в1", "в2", "в3"):
        agent.send(message)
    assert llm.messages_of(2) == [system(), user("в2"), bot("а2"), user("в3")]


def test_r6_history_unchanged_on_error():
    agent, _ = make_agent([text("а1"), ConnectionError("сбой API")])
    agent.send("в1")
    raises(ConnectionError, agent.send, "в2")
    assert agent.history == [user("в1"), bot("а1")]


def test_r6_can_continue_after_error():
    agent, llm = make_agent([ConnectionError("сбой API"), text("ок")])
    raises(ConnectionError, agent.send, "первая попытка")
    assert agent.send("снова") == "ок"
    assert llm.messages_of(1) == [system(), user("снова")]


def summary_scenario():
    agent, llm = make_agent(
        [text("ответ 1"), text("ответ 2"), text("ответ 3"), text("РЕЗЮМЕ"), text("ответ 4")],
        summarize_threshold=4,
    )
    for message in ("один", "два", "три", "четыре"):
        agent.send(message)
    return agent, llm


def test_r7_summary_call():
    agent, llm = summary_scenario()
    assert llm.call_count == 5
    request = llm.messages_of(3)
    assert request[0] == {"role": "system", "content": sol.SUMMARY_PROMPT}
    transcript = request[1]["content"]
    for fragment in ("один", "ответ 1", "два", "ответ 2"):
        assert fragment in transcript, f"в текст для пересказа не попало «{fragment}»"
    assert "три" not in transcript, "последние KEEP_LAST сообщений не нужно пересказывать"
    assert agent.summary == "РЕЗЮМЕ"


def test_r7_history_after_summary():
    agent, _ = summary_scenario()
    assert agent.history == [user("три"), bot("ответ 3"), user("четыре"), bot("ответ 4")]


def test_r7_no_summary_below_threshold():
    agent, llm = make_agent([text("а1"), text("а2"), text("а3")], summarize_threshold=4)
    for message in ("в1", "в2", "в3"):
        agent.send(message)
    assert llm.call_count == 3 and agent.summary is None


def test_r8_summary_in_context():
    _, llm = summary_scenario()
    assert llm.messages_of(4) == [
        system(),
        {"role": "system", "content": sol.SUMMARY_PREFIX + "РЕЗЮМЕ"},
        user("три"),
        bot("ответ 3"),
        user("четыре"),
    ]


def test_r8_previous_summary_included():
    agent, llm = make_agent(
        [text("а1"), text("а2"), text("S1"), text("а3"), text("S2"), text("а4")],
        summarize_threshold=2,
    )
    for message in ("один", "два", "три", "четыре"):
        agent.send(message)
    second_summary_request = llm.messages_of(4)[1]["content"]
    assert "S1" in second_summary_request, "старое резюме нужно включить в новый пересказ"
    assert "два" in second_summary_request
    assert agent.summary == "S2"


def test_r8_reset_clears_summary():
    agent, llm = summary_scenario()
    llm.responses.append(text("новый диалог"))
    agent.reset()
    agent.send("привет")
    assert agent.summary is None
    assert llm.messages_of(5) == [system(), user("привет")]

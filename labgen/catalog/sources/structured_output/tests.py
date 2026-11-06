import json

import solution as sol
from agentkit import ScriptedLLM

TEXT = "Здравствуйте! Хочу заказать 2 ноутбука с доставкой в Казань."
ORDER = {"product": "ноутбук", "quantity": 2, "city": "Казань"}
ORDER_JSON = json.dumps(ORDER, ensure_ascii=False)


def raises(exc_type, func, *args, **kwargs):
    """Проверяет, что вызов выбрасывает исключение нужного типа, и возвращает его."""
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"ожидалось исключение {exc_type.__name__}")


def test_r1_messages_structure():
    messages = sol.build_messages(TEXT)
    assert isinstance(messages, list) and len(messages) == 2, "нужно ровно два сообщения"
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": TEXT}


def test_r1_system_prompt_mentions_json_and_fields():
    messages = sol.build_messages(TEXT)
    assert messages[0]["content"] == sol.SYSTEM_PROMPT
    prompt = sol.SYSTEM_PROMPT.lower()
    assert "json" in prompt, "в системной инструкции нужно потребовать JSON"
    for name in ("product", "quantity", "city"):
        assert name in prompt, f"в инструкции не названо поле {name}"


def test_r2_parse_plain_json():
    assert sol.parse_response(ORDER_JSON) == ORDER


def test_r2_parse_json_with_whitespace():
    content = '\n  {"product": "чай", "quantity": 1, "city": "Омск"}\n'
    assert sol.parse_response(content) == {"product": "чай", "quantity": 1, "city": "Омск"}


def test_r3_single_call_and_result():
    llm = ScriptedLLM([ORDER_JSON])
    assert sol.extract_order(llm, TEXT) == ORDER
    assert llm.call_count == 1, "при корректном ответе нужен ровно один вызов модели"


def test_r3_sends_built_messages():
    llm = ScriptedLLM([ORDER_JSON])
    sol.extract_order(llm, TEXT)
    assert llm.messages_of(0) == sol.build_messages(TEXT)


def test_r4_json_in_code_fence():
    content = "Вот результат:\n```json\n" + ORDER_JSON + "\n```"
    assert sol.parse_response(content) == ORDER


def test_r4_json_inside_text():
    content = "Конечно! " + ORDER_JSON + " Обращайтесь ещё."
    assert sol.parse_response(content) == ORDER


def test_r4_invalid_json_raises():
    raises(sol.ExtractionError, sol.parse_response, "это не JSON")
    raises(sol.ExtractionError, sol.parse_response, '{"product": "чай", quantity: 1}')
    raises(sol.ExtractionError, sol.parse_response, "")


def test_r5_missing_fields_listed():
    error = raises(sol.ExtractionError, sol.validate_order, {"product": "чай"})
    assert "quantity" in str(error) and "city" in str(error), "в сообщении нужно перечислить недостающие поля"


def test_r5_empty_values_are_missing():
    error = raises(sol.ExtractionError, sol.validate_order, {"product": "", "quantity": 1, "city": None})
    assert "product" in str(error) and "city" in str(error)


def test_r6_quantity_converted_and_strings_stripped():
    data = {"product": " чай ", "quantity": "3", "city": "Омск "}
    assert sol.validate_order(data) == {"product": "чай", "quantity": 3, "city": "Омск"}


def test_r6_bad_quantity():
    for quantity in ("много", 0, -2):
        raises(sol.ExtractionError, sol.validate_order, {"product": "чай", "quantity": quantity, "city": "Омск"})


def test_r7_retry_after_invalid_json():
    llm = ScriptedLLM(["не знаю", ORDER_JSON])
    assert sol.extract_order(llm, TEXT) == ORDER
    assert llm.call_count == 2
    second = llm.messages_of(1)
    assert len(second) == 4, "во второй запрос нужно добавить ответ модели и сообщение об ошибке"
    assert second[2] == {"role": "assistant", "content": "не знаю"}
    assert second[3]["role"] == "user"


def test_r7_error_description_sent():
    llm = ScriptedLLM(['{"product": "чай", "quantity": 1}', ORDER_JSON])
    sol.extract_order(llm, TEXT)
    assert "city" in llm.messages_of(1)[-1]["content"], "модели нужно сообщить, что именно не так"


def test_r7_retry_limit():
    llm = ScriptedLLM(["плохо", "снова плохо", ORDER_JSON])
    raises(sol.ExtractionError, sol.extract_order, llm, TEXT, max_retries=1)
    assert llm.call_count == 2, "при max_retries=1 допустим один повтор"


def test_r8_raises_after_retries():
    llm = ScriptedLLM(["нет", "нет", "нет"])
    raises(sol.ExtractionError, sol.extract_order, llm, TEXT)
    assert llm.call_count == 3


def test_r8_llm_failure_not_masked():
    llm = ScriptedLLM([ConnectionError("API недоступен")])
    raises(ConnectionError, sol.extract_order, llm, TEXT)

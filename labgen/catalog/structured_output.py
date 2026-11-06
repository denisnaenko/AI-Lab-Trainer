from .base import MutantDef, RequirementDef, TopicDef

TOPIC = TopicDef(
    id="structured_output",
    title="Структурированный вывод LLM",
    summary="Промпт-инструкция, разбор JSON-ответа модели, валидация полей и повторные запросы.",
    keywords=(
        "структурированный", "структурированного", "json", "вывод", "извлечение", "парсинг",
        "промпт", "prompt", "structured", "output", "ответ модели", "схема", "валидация", "шаблон промпта",
    ),
    context_md="""
Интернет-магазин получает от клиентов сообщения в свободной форме: «Хочу 2 ноутбука с доставкой в Казань».
Чтобы оформить заказ автоматически, из сообщения нужно извлечь три поля: товар (`product`),
количество (`quantity`) и город доставки (`city`).

Извлечение делает LLM. Ваша задача — построить надёжную обвязку вокруг модели: правильно
сформулировать запрос, разобрать ответ, проверить данные и обработать ситуации, когда модель
отвечает не по формату.
""",
    interface_md={
        1: """
Реализуйте в `solution.py`:

- `SYSTEM_PROMPT` — системная инструкция для модели;
- `build_messages(text)` — список сообщений для модели;
- `parse_response(content)` — разбор JSON-ответа модели в словарь;
- `extract_order(llm, text)` — полный цикл: сообщения → вызов модели → словарь заказа.
""",
        2: """
Реализуйте в `solution.py` функции `build_messages`, `parse_response`, `validate_order`,
`extract_order` и исключение `ExtractionError`. Модель может обернуть JSON в блок кода
или добавить к нему пояснения — разбор должен с этим справляться, а данные заказа нужно проверять.
""",
        3: """
Реализуйте в `solution.py` функции `build_messages`, `parse_response`, `validate_order`,
`extract_order(llm, text, max_retries=2)` и исключение `ExtractionError`.
Модель иногда отвечает не по формату — агент должен сообщить ей об ошибке и попросить исправиться.
""",
    },
    requirements=[
        RequirementDef("R1", 1, "`build_messages(text)` возвращает два сообщения: системное с `SYSTEM_PROMPT` (в нём требуется ответ в JSON с полями product, quantity и city) и пользовательское с исходным текстом."),
        RequirementDef("R2", 1, "`parse_response(content)` превращает JSON-ответ модели в словарь."),
        RequirementDef("R3", 1, "`extract_order(llm, text)` отправляет модели сообщения из `build_messages`, при корректном ответе делает ровно один вызов `llm.chat` и возвращает словарь заказа."),
        RequirementDef("R4", 2, "`parse_response` находит JSON внутри блока ```` ```json ```` и внутри текста; если JSON нет или он некорректен, выбрасывает `ExtractionError`."),
        RequirementDef("R5", 2, "`validate_order(data)` проверяет наличие полей product, quantity и city (пустая строка и `None` считаются отсутствием) и выбрасывает `ExtractionError` с перечнем недостающих полей."),
        RequirementDef("R6", 2, "`validate_order` приводит quantity к `int` и обрезает пробелы в строковых полях; нечисловое или неположительное quantity приводит к `ExtractionError`."),
        RequirementDef("R7", 3, "При некорректном ответе `extract_order` повторяет запрос: добавляет в диалог ответ модели и сообщение пользователя с описанием ошибки; повторов не больше `max_retries`."),
        RequirementDef("R8", 3, "Если корректного ответа нет и после всех повторов, `extract_order` выбрасывает `ExtractionError`; исключения самой модели (сбой API) не подменяются."),
    ],
    public_tests={
        "test_r1_messages_structure",
        "test_r2_parse_plain_json",
        "test_r3_single_call_and_result",
        "test_r4_json_in_code_fence",
        "test_r5_missing_fields_listed",
        "test_r6_quantity_converted_and_strings_stripped",
        "test_r7_retry_after_invalid_json",
        "test_r8_raises_after_retries",
    },
    mutants=[
        MutantDef("M1", 1, "Системное сообщение не отправляется",
                  '        {"role": "system", "content": SYSTEM_PROMPT},\n', ""),
        MutantDef("M2", 1, "В модель уходит пустой текст клиента",
                  '{"role": "user", "content": text}', '{"role": "user", "content": ""}'),
        MutantDef("M3", 1, "Ответ модели не разбирается",
                  "            return validate_order(parse_response(response.content))",
                  '            return {"raw": response.content}'),
        MutantDef("M4", 2, "JSON внутри текста не находится",
                  '        start, end = content.find("{"), content.rfind("}")',
                  "        start, end = 0, len(content) - 1"),
        MutantDef("M5", 2, "Некорректный JSON не превращается в ExtractionError",
                  "    except json.JSONDecodeError as exc:", "    except KeyError as exc:"),
        MutantDef("M6", 2, "Обязательные поля не проверяются", "    if missing:\n", "    if False:\n"),
        MutantDef("M7", 2, "quantity не приводится к числу",
                  '        "quantity": quantity,', '        "quantity": data["quantity"],'),
        MutantDef("M8", 2, "Нулевое количество допускается", "    if quantity <= 0:", "    if quantity < 0:"),
        MutantDef("M9", 3, "Нет повторных запросов",
                  "    for _ in range(max_retries + 1):", "    for _ in range(1):"),
        MutantDef("M10", 3, "Модели не сообщают, в чём ошибка",
                  '{"role": "user", "content": f"Ошибка: {exc}. Исправь ответ и верни только корректный JSON."}',
                  '{"role": "user", "content": "Попробуй ещё раз."}'),
        MutantDef("M11", 3, "Лишний повторный запрос",
                  "    for _ in range(max_retries + 1):", "    for _ in range(max_retries + 2):"),
    ],
    hints={
        1: [
            "Сообщение — это словарь с ключами `role` и `content`.",
            "Чем точнее инструкция описывает формат ответа, тем реже модель от него отклоняется.",
        ],
        2: [
            "Для блока кода удобно регулярное выражение, для JSON внутри текста — поиск первой `{` и последней `}`.",
            "`json.JSONDecodeError` стоит перехватить и превратить в `ExtractionError` с понятным текстом.",
        ],
        3: [
            "Не изменяйте исходный список сообщений «на месте» — так проще отлаживать повторы.",
            "Сбой модели (исключение из `llm.chat`) и неверный формат ответа — разные ситуации.",
        ],
    },
    minutes={1: 40, 2: 80, 3: 120},
)

"""Веб-интерфейс: вход, интеграции, генерация в фоне, рабочее пространство, права доступа."""

import pytest


def create_lab(browser, service, topic="Создание агента с инструментами", level="1", source="catalog"):
    response = browser.post("/labs", {"topic": topic, "level": level, "source": source, "topic_id": "",
                                      "materials": "", "simulate": "0"}, follow_redirects=False)
    assert response.status_code == 303, response.text
    job_id = response.headers["location"].split("/")[-1]
    job = service.jobs.wait(job_id)
    assert job.status == "done", job.error
    return job.lab_id


def test_pages_require_login(browser):
    assert "Начать бесплатно" in browser.get("/").text
    response = browser.get("/labs/new", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"].startswith("/login")
    assert browser.client.get("/api/labs").status_code == 401


def test_csrf_is_required(browser):
    response = browser.client.post("/register", data={"email": "x@y.ru", "password": "password1", "password2": "password1"})
    assert response.status_code == 403


def test_login_flow(browser, make_browser):
    browser.register(email="anna@urfu.ru", password="password123")
    other = make_browser()
    bad = other.post("/login", {"email": "anna@urfu.ru", "password": "nope-nope"})
    assert bad.status_code == 400 and "Неверная почта" in bad.text
    ok = other.post("/login", {"email": "anna@urfu.ru", "password": "password123", "next": "/account"}, follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/account"


def test_llm_integrations_in_account(browser, service):
    browser.register()
    page = browser.post("/account/llm/anthropic", {"api_key": "sk-ant-api03-SECRET-1234", "model": "claude-sonnet-5-5"},
                        follow_redirects=True)
    assert "sk-…1234" in page.text and "SECRET" not in page.text
    browser.post("/account/llm/local", {"model": "qwen2.5-coder:14b", "base_url": "http://127.0.0.1:9/v1"})
    result = browser.post_json("/api/account/llm/local/test").json()
    assert result["ok"] is False
    assert "ошибка подключения" in browser.get("/account").text
    browser.post("/account/llm/local/default")
    user_id = service.accounts.authenticate("teacher@urfu.ru", "password123").id
    assert service.accounts.default_provider(user_id) == "local"
    assert "Локальная LLM · qwen2.5-coder:14b" in browser.get("/").text


def test_full_teacher_flow(browser, service):
    browser.register()
    lab_id = create_lab(browser, service)
    page = browser.get(f"/labs/{lab_id}")
    assert page.status_code == 200 and "lab-data" in page.text

    spec = service.store.get(lab_id).spec
    run = browser.post_json(f"/api/labs/{lab_id}/run", {"code": spec.starter_code}).json()
    assert run["passed"] == 0 and run["total"] > 0
    run = browser.post_json(f"/api/labs/{lab_id}/run", {"code": spec.reference_solution}).json()
    assert run["passed"] == run["total"]
    submit = browser.post_json(f"/api/labs/{lab_id}/submit", {"code": spec.reference_solution}).json()
    assert submit["auto_score"] == submit["auto_max"]
    assert "Принято" in browser.get(submit["url"]).text

    # редактирование: новая версия и повторная проверка
    record = service.store.get(lab_id)
    form = {
        "title": "Агент-синоптик", "summary": record.spec.summary, "estimated_minutes": "50",
        "statement_md": record.spec.statement_md, "starter_code": record.spec.starter_code,
        "reference_solution": record.spec.reference_solution, "tests_public": record.spec.tests_public,
        "tests_hidden": record.spec.tests_hidden, "mutants_json": "[]",
        "req_count": str(len(record.spec.requirements)), "note": "переименовал",
    }
    for i, r in enumerate(record.spec.requirements):
        form.update({f"req_id_{i}": r.id, f"req_text_{i}": r.text, f"req_points_{i}": str(r.points)})
        if r.auto:
            form[f"req_auto_{i}"] = "on"
    assert browser.post(f"/labs/{lab_id}/edit", form, follow_redirects=False).status_code == 303
    updated = service.store.get(lab_id)
    assert updated.spec.title == "Агент-синоптик" and updated.version == 2 and updated.validation.passed

    browser.post(f"/labs/{lab_id}/approve")
    assert service.store.get(lab_id).status == "approved"
    for kind in ("student", "teacher", "moodle"):
        response = browser.get(f"/labs/{lab_id}/export/{kind}")
        assert response.status_code == 200 and response.content[:2] == b"PK"

    browser.post(f"/labs/{lab_id}/restore/1")
    restored = service.store.get(lab_id)
    assert restored.spec.title == "Агент с инструментами" and restored.version == 3


def test_labs_are_private_between_users(browser, make_browser, service):
    browser.register(email="a@urfu.ru")
    lab_id = create_lab(browser, service)
    other = make_browser()
    other.register(email="b@urfu.ru")
    assert other.get(f"/labs/{lab_id}").status_code == 404
    assert other.post_json(f"/api/labs/{lab_id}/submit", {"code": ""}).status_code == 404
    assert lab_id not in other.get("/").text


def test_shared_catalog_labs_visible_to_all(browser, service):
    record = service.create("RAG", 1, mode="catalog", topic_id="rag_agent")  # без владельца — общая
    browser.register()
    assert record.id in browser.get("/").text
    assert browser.get(f"/labs/{record.id}").status_code == 200


def test_broken_edit_blocks_approval(browser, service):
    browser.register()
    lab_id = create_lab(browser, service, "Агент с памятью")
    service.update(lab_id, {"reference_solution": "class ChatAgent:\n    pass\n"})
    browser.post(f"/labs/{lab_id}/approve")
    assert service.store.get(lab_id).status == "draft"


def test_unknown_topic_job_reports_error(browser, service):
    browser.register()
    response = browser.post("/labs", {"topic": "История Древнего Рима", "level": "1", "source": "catalog"},
                            follow_redirects=False)
    job_id = response.headers["location"].split("/")[-1]
    job = service.jobs.wait(job_id)
    assert job.status == "error" and "не найдена" in job.error
    assert browser.get(f"/api/jobs/{job_id}").json()["status"] == "error"


def test_llm_source_without_integration(browser):
    browser.register()
    response = browser.post("/labs", {"topic": "Агент", "level": "1", "source": "openai"}, follow_redirects=True)
    assert "не настроена" in response.text


@pytest.mark.parametrize("path", ["/catalog", "/login", "/register"])
def test_public_pages(browser, path):
    assert browser.get(path).status_code == 200

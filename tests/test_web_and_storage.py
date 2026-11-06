from fastapi.testclient import TestClient

from labgen.web.app import create_app


def make_client(service):
    return TestClient(create_app(service))


def create_lab(client, topic="Создание агента с инструментами", level="1"):
    response = client.post("/labs", data={"topic": topic, "level": level, "mode": "catalog"}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return response.headers["location"].split("/")[-1]


def test_full_teacher_flow(service):
    client = make_client(service)
    assert client.get("/").status_code == 200
    lab_id = create_lab(client)
    page = client.get(f"/labs/{lab_id}")
    assert page.status_code == 200 and "Агент с инструментами" in page.text

    # редактирование: новая версия, повторная проверка, статус «черновик»
    record = service.store.get(lab_id)
    form = {
        "title": "Агент-синоптик",
        "summary": record.spec.summary,
        "estimated_minutes": "50",
        "statement_md": record.spec.statement_md.replace("Агент с инструментами", "Агент-синоптик"),
        "starter_code": record.spec.starter_code,
        "reference_solution": record.spec.reference_solution,
        "tests_public": record.spec.tests_public,
        "tests_hidden": record.spec.tests_hidden,
        "mutants_json": "[]",
        "req_count": str(len(record.spec.requirements)),
        "note": "переименовал",
    }
    for i, r in enumerate(record.spec.requirements):
        form.update({f"req_id_{i}": r.id, f"req_text_{i}": r.text, f"req_points_{i}": str(r.points)})
        if r.auto:
            form[f"req_auto_{i}"] = "on"
    response = client.post(f"/labs/{lab_id}/edit", data=form, follow_redirects=False)
    assert response.status_code == 303
    updated = service.store.get(lab_id)
    assert updated.spec.title == "Агент-синоптик" and updated.version == 2
    assert updated.validation.passed
    assert service.store.versions(lab_id) == [1, 2]

    # утверждение, экспорт, проверка решения
    client.post(f"/labs/{lab_id}/approve")
    assert service.store.get(lab_id).status == "approved"
    for kind in ("student", "teacher", "moodle"):
        response = client.get(f"/labs/{lab_id}/export/{kind}")
        assert response.status_code == 200 and response.content[:2] == b"PK"
    response = client.post(f"/labs/{lab_id}/grade", data={"student": "Иванов", "code": updated.spec.reference_solution})
    assert response.status_code == 200 and "Результат автопроверки" in response.text

    # восстановление версии
    client.post(f"/labs/{lab_id}/restore/1")
    restored = service.store.get(lab_id)
    assert restored.spec.title == "Агент с инструментами" and restored.version == 3


def test_broken_edit_blocks_approval(service):
    client = make_client(service)
    lab_id = create_lab(client, "Агент с памятью", "1")
    service.update(lab_id, {"reference_solution": "class ChatAgent:\n    pass\n"})
    response = client.post(f"/labs/{lab_id}/approve", follow_redirects=True)
    assert "Нельзя утвердить" in response.text
    assert service.store.get(lab_id).status == "draft"


def test_unknown_topic_shows_error(service):
    client = make_client(service)
    response = client.post("/labs", data={"topic": "История Древнего Рима", "level": "1", "mode": "catalog"})
    assert response.status_code == 400 and "не найдена" in response.text


def test_api(service):
    client = make_client(service)
    created = client.post("/api/labs", json={"topic": "RAG по базе знаний", "level": 2, "mode": "catalog"})
    assert created.status_code == 201
    lab_id = created.json()["id"]
    assert client.get("/api/labs").json()[0]["id"] == lab_id
    spec = service.store.get(lab_id).spec
    graded = client.post(f"/api/labs/{lab_id}/grade", json={"code": spec.reference_solution, "student": "api"})
    assert graded.json()["auto_score"] == 90
    assert client.get("/api/labs/does-not-exist").status_code == 404

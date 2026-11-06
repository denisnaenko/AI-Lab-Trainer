import re

import pytest
from fastapi.testclient import TestClient

from labgen.config import load_settings
from labgen.service import LabService
from labgen.storage import LabStore


@pytest.fixture
def settings(tmp_path):
    s = load_settings(env_file=None)
    s.data_dir = tmp_path / "data"
    s.secret_key = "test-secret"
    s.llm_provider = ""
    s.llm_base_url = ""
    s.llm_model = ""
    return s


@pytest.fixture
def service(settings):
    return LabService(settings, store=LabStore(settings.data_dir), use_settings_llm=False)


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


class Browser:
    """Тестовый клиент с сессией и CSRF-токеном, как у браузера."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.csrf = csrf_of(client.get("/login").text)

    def refresh_csrf(self, path: str = "/") -> None:
        self.csrf = csrf_of(self.client.get(path).text)

    def post(self, url: str, data: dict | None = None, **kwargs):
        return self.client.post(url, data={"csrf": self.csrf, **(data or {})}, **kwargs)

    def post_json(self, url: str, payload: dict | None = None):
        return self.client.post(url, json=payload or {}, headers={"X-CSRF-Token": self.csrf})

    def get(self, url: str, **kwargs):
        return self.client.get(url, **kwargs)

    def register(self, email: str = "teacher@urfu.ru", password: str = "password123", name: str = "Преподаватель"):
        response = self.post("/register", {"name": name, "email": email, "password": password, "password2": password},
                             follow_redirects=False)
        assert response.status_code == 303, response.text
        self.refresh_csrf()
        return response


@pytest.fixture
def app(service):
    from labgen.web.app import create_app

    return create_app(service)


@pytest.fixture
def browser(app):
    return Browser(TestClient(app))


@pytest.fixture
def make_browser(app):
    return lambda: Browser(TestClient(app))

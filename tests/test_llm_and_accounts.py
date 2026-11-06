"""Интеграции LLM через LangChain и аккаунты: шифрование ключей, пароли, модели по умолчанию."""

import sqlite3

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from labgen.accounts import AccountError, AccountStore, verify_password
from labgen.llm import PROVIDERS, LabLLM, LLMConfig, LLMError, build_chat_model, check_connection, fake_llm


def test_factory_builds_langchain_models():
    claude = build_chat_model(LLMConfig("anthropic", "claude-sonnet-5-5", api_key="sk-ant-x"))
    assert isinstance(claude, ChatAnthropic) and claude.model == "claude-sonnet-5-5"
    openai = build_chat_model(LLMConfig("openai", "gpt-5-mini", api_key="sk-x"))
    assert isinstance(openai, ChatOpenAI) and openai.model_name == "gpt-5-mini"
    local = build_chat_model(LLMConfig("local", "qwen2.5-coder:14b", base_url="http://localhost:11434/v1/"))
    assert isinstance(local, ChatOpenAI) and local.openai_api_base == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "config",
    [
        LLMConfig("anthropic", "claude-sonnet-5-5"),
        LLMConfig("openai", "gpt-5-mini"),
        LLMConfig("local", "qwen"),
        LLMConfig("gigachat", "x"),
    ],
)
def test_factory_validates_config(config):
    with pytest.raises(LLMError):
        build_chat_model(config)


def test_check_connection_reports_unreachable_server():
    ok, message = check_connection(LLMConfig("local", "qwen", base_url="http://127.0.0.1:9/v1"))
    assert not ok and message


def test_lab_llm_records_messages():
    llm = fake_llm(["ответ"])
    assert llm.complete([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]) == "ответ"
    assert llm.chat.calls == [[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]]
    assert isinstance(llm, LabLLM) and llm.usage["requests"] == 1


@pytest.fixture
def accounts(tmp_path):
    return AccountStore(tmp_path / "app.db", "secret-1")


def test_user_registration_and_login(accounts):
    user = accounts.create_user("Teacher@URFU.ru", "Анна", "password123")
    assert user.email == "teacher@urfu.ru"
    assert accounts.authenticate("teacher@urfu.ru", "password123").id == user.id
    assert accounts.authenticate("teacher@urfu.ru", "wrong-pass") is None
    with pytest.raises(AccountError):
        accounts.create_user("teacher@urfu.ru", "Дубль", "password123")
    with pytest.raises(AccountError):
        accounts.create_user("bad-email", "x", "password123")
    with pytest.raises(AccountError):
        accounts.create_user("a@b.ru", "x", "short")


def test_password_is_hashed(accounts, tmp_path):
    accounts.create_user("t@urfu.ru", "T", "password123")
    stored = sqlite3.connect(tmp_path / "app.db").execute("SELECT password_hash FROM users").fetchone()[0]
    assert "password123" not in stored and stored.startswith("scrypt$")
    assert verify_password("password123", stored)


def test_api_keys_encrypted_and_masked(accounts, tmp_path):
    user = accounts.create_user("t@urfu.ru", "T", "password123")
    key = "sk-ant-api03-VERYSECRETVALUE-9876"
    cred = accounts.save_credential(user.id, "anthropic", api_key=key, model="claude-sonnet-5-5")
    raw = sqlite3.connect(tmp_path / "app.db").execute("SELECT api_key_enc FROM llm_credentials").fetchone()[0]
    assert key not in raw and "VERYSECRET" not in raw
    assert cred.key_mask == "sk-…9876" and cred.is_default
    assert accounts.llm_config(user.id).api_key == key
    # ключ не меняется, если поле оставили пустым
    accounts.save_credential(user.id, "anthropic", api_key=None, model="claude-opus-5-5")
    assert accounts.llm_config(user.id, "anthropic").api_key == key
    # другой секрет приложения — ключ не расшифровать
    other = AccountStore(tmp_path / "app.db", "secret-2")
    assert other.llm_config(user.id, "anthropic").api_key == ""


def test_default_provider_switching(accounts):
    user = accounts.create_user("t@urfu.ru", "T", "password123")
    accounts.save_credential(user.id, "openai", api_key="sk-1", model="gpt-5-mini")
    accounts.save_credential(user.id, "local", api_key=None, model="qwen", base_url="http://localhost:11434/v1")
    assert accounts.default_provider(user.id) == "openai"
    accounts.set_default(user.id, "local")
    creds = accounts.credentials(user.id)
    assert creds["local"].is_default and not creds["openai"].is_default
    accounts.delete_credential(user.id, "local")
    assert accounts.default_provider(user.id) == "openai"
    assert accounts.credentials(user.id)["openai"].is_default


def test_credential_validation(accounts):
    user = accounts.create_user("t@urfu.ru", "T", "password123")
    with pytest.raises(AccountError):
        accounts.save_credential(user.id, "openai", api_key="", model="gpt-5-mini")
    with pytest.raises(AccountError):
        accounts.save_credential(user.id, "local", api_key=None, model="qwen", base_url="localhost:11434")
    assert set(PROVIDERS) == {"anthropic", "openai", "local"}

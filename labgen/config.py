"""Настройки приложения из переменных окружения (и необязательного файла .env)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Минимальный загрузчик .env: строки KEY=VALUE, комментарии с #."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    data_dir: Path
    secret_key: str
    # LLM по умолчанию для CLI; в веб-интерфейсе у каждого пользователя свои интеграции
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_json_mode: bool
    llm_timeout: float
    llm_temperature: float | None
    max_generation_attempts: int
    simulate_students: int
    allow_registration: bool = True

    def resolve_secret(self) -> str:
        """Секрет для сессий и шифрования ключей: из окружения или из файла в каталоге данных."""
        if self.secret_key:
            return self.secret_key
        path = self.data_dir / "secret.key"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        value = secrets.token_urlsafe(48)
        path.write_text(value, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return value


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    if env_file:
        _load_dotenv(Path(env_file))
    env = os.environ.get
    temperature = env("LABGEN_LLM_TEMPERATURE", "").strip()
    return Settings(
        data_dir=Path(env("LABGEN_DATA_DIR", "data")),
        secret_key=env("LABGEN_SECRET_KEY", ""),
        llm_provider=env("LABGEN_LLM_PROVIDER", "").strip().lower(),
        llm_base_url=env("LABGEN_LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=env("LABGEN_LLM_API_KEY", ""),
        llm_model=env("LABGEN_LLM_MODEL", ""),
        llm_json_mode=env("LABGEN_LLM_JSON_MODE", "1") not in ("0", "false", "no"),
        llm_timeout=float(env("LABGEN_LLM_TIMEOUT", "240")),
        llm_temperature=float(temperature) if temperature else None,
        max_generation_attempts=int(env("LABGEN_MAX_ATTEMPTS", "3")),
        simulate_students=int(env("LABGEN_SIMULATE_STUDENTS", "0")),
        allow_registration=env("LABGEN_ALLOW_REGISTRATION", "1") not in ("0", "false", "no"),
    )

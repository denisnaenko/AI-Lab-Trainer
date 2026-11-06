"""Настройки приложения из переменных окружения (и необязательного файла .env)."""

from __future__ import annotations

import os
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
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_json_mode: bool
    llm_timeout: float
    llm_temperature: float
    max_generation_attempts: int
    simulate_students: int

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    if env_file:
        _load_dotenv(Path(env_file))
    env = os.environ.get
    return Settings(
        data_dir=Path(env("LABGEN_DATA_DIR", "data")),
        llm_base_url=env("LABGEN_LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=env("LABGEN_LLM_API_KEY", ""),
        llm_model=env("LABGEN_LLM_MODEL", ""),
        llm_json_mode=env("LABGEN_LLM_JSON_MODE", "1") not in ("0", "false", "no"),
        llm_timeout=float(env("LABGEN_LLM_TIMEOUT", "180")),
        llm_temperature=float(env("LABGEN_LLM_TEMPERATURE", "0.2")),
        max_generation_attempts=int(env("LABGEN_MAX_ATTEMPTS", "3")),
        simulate_students=int(env("LABGEN_SIMULATE_STUDENTS", "0")),
    )

"""Пользователи и их интеграции с LLM.

* Пароли хранятся как хеш scrypt с солью.
* API-ключи шифруются (Fernet, ключ шифрования выводится из секрета приложения)
  и расшифровываются только в момент обращения к модели. В интерфейс ключ
  возвращается лишь в виде маски «sk-…abcd».
* Хранилище — SQLite в каталоге данных.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .llm.factory import PROVIDERS, LLMConfig
from .models import now_iso

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_credentials (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    api_key_enc TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    base_url TEXT NOT NULL DEFAULT '',
    json_mode INTEGER NOT NULL DEFAULT 0,
    is_default INTEGER NOT NULL DEFAULT 0,
    last_check_ok INTEGER,
    last_check_message TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, provider)
);
"""


class AccountError(ValueError):
    pass


@dataclass
class User:
    id: int
    email: str
    name: str
    created_at: str

    @property
    def initials(self) -> str:
        return (self.name or self.email)[:1].upper()


@dataclass
class Credential:
    provider: str
    model: str
    base_url: str
    json_mode: bool
    is_default: bool
    key_mask: str
    has_key: bool
    last_check_ok: bool | None
    last_check_message: str | None
    updated_at: str

    @property
    def title(self) -> str:
        return PROVIDERS[self.provider].title

    @property
    def label(self) -> str:
        return f"{self.title} · {self.model}"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, digest_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(digest, expected)


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "••••"
    return f"{key[:3]}…{key[-4:]}"


class AccountStore:
    def __init__(self, path: Path | str, secret: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fernet_key = base64.urlsafe_b64encode(hashlib.sha256(("labgen-keys:" + secret).encode()).digest())
        self._fernet = Fernet(fernet_key)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # --- пользователи -------------------------------------------------------------

    def create_user(self, email: str, name: str, password: str) -> User:
        email = email.strip().lower()
        name = name.strip() or email.split("@")[0]
        if not EMAIL_RE.match(email):
            raise AccountError("Укажите корректный адрес почты")
        if len(password) < MIN_PASSWORD:
            raise AccountError(f"Пароль должен быть не короче {MIN_PASSWORD} символов")
        try:
            with self._connect() as db:
                cursor = db.execute(
                    "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (email, name[:80], hash_password(password), now_iso()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise AccountError("Пользователь с такой почтой уже зарегистрирован") from None
        return self.get_user(user_id)  # type: ignore[return-value]

    def authenticate(self, email: str, password: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        if row is None:
            hash_password(password)  # одинаковое время ответа для существующей и несуществующей почты
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return User(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])

    def get_user(self, user_id: int) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return User(id=row["id"], email=row["email"], name=row["name"], created_at=row["created_at"])

    def update_profile(self, user_id: int, name: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE users SET name = ? WHERE id = ?", (name.strip()[:80], user_id))

    def change_password(self, user_id: int, old: str, new: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None or not verify_password(old, row["password_hash"]):
                raise AccountError("Текущий пароль указан неверно")
            if len(new) < MIN_PASSWORD:
                raise AccountError(f"Пароль должен быть не короче {MIN_PASSWORD} символов")
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), user_id))

    # --- интеграции LLM -------------------------------------------------------------

    def _encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode() if value else ""

    def _decrypt(self, value: str) -> str:
        if not value:
            return ""
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except InvalidToken:
            return ""  # секрет приложения сменился — ключ придётся ввести заново

    def save_credential(
        self,
        user_id: int,
        provider: str,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "",
        json_mode: bool = False,
    ) -> Credential:
        """``api_key=None`` — оставить сохранённый ключ без изменений."""
        info = PROVIDERS.get(provider)
        if info is None:
            raise AccountError("Неизвестный провайдер")
        model = model.strip() or info.default_model
        base_url = base_url.strip().rstrip("/")
        if info.needs_url:
            if not re.match(r"^https?://", base_url):
                raise AccountError("Адрес сервера должен начинаться с http:// или https://")
        else:
            base_url = ""
        existing = self._row(user_id, provider)
        if api_key is None:
            key_enc = existing["api_key_enc"] if existing else ""
        else:
            key_enc = self._encrypt(api_key.strip())
        if info.needs_key and not key_enc:
            raise AccountError("Укажите API-ключ")
        with self._connect() as db:
            has_default = db.execute(
                "SELECT 1 FROM llm_credentials WHERE user_id = ? AND is_default = 1 AND provider != ?", (user_id, provider)
            ).fetchone()
            is_default = bool(existing["is_default"]) if existing else not has_default
            db.execute(
                """INSERT INTO llm_credentials (user_id, provider, api_key_enc, model, base_url, json_mode, is_default,
                       last_check_ok, last_check_message, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                   ON CONFLICT(user_id, provider) DO UPDATE SET api_key_enc = excluded.api_key_enc,
                       model = excluded.model, base_url = excluded.base_url, json_mode = excluded.json_mode,
                       last_check_ok = NULL, last_check_message = NULL, updated_at = excluded.updated_at""",
                (user_id, provider, key_enc, model, base_url, int(json_mode), int(is_default), now_iso()),
            )
        return self.credential(user_id, provider)  # type: ignore[return-value]

    def _row(self, user_id: int, provider: str):
        with self._connect() as db:
            return db.execute(
                "SELECT * FROM llm_credentials WHERE user_id = ? AND provider = ?", (user_id, provider)
            ).fetchone()

    def _to_credential(self, row) -> Credential:
        key = self._decrypt(row["api_key_enc"])
        return Credential(
            provider=row["provider"],
            model=row["model"],
            base_url=row["base_url"],
            json_mode=bool(row["json_mode"]),
            is_default=bool(row["is_default"]),
            key_mask=mask_key(key),
            has_key=bool(key),
            last_check_ok=None if row["last_check_ok"] is None else bool(row["last_check_ok"]),
            last_check_message=row["last_check_message"],
            updated_at=row["updated_at"],
        )

    def credential(self, user_id: int, provider: str) -> Credential | None:
        row = self._row(user_id, provider)
        return self._to_credential(row) if row else None

    def credentials(self, user_id: int) -> dict[str, Credential]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM llm_credentials WHERE user_id = ?", (user_id,)).fetchall()
        result = {row["provider"]: self._to_credential(row) for row in rows}
        return {p: result[p] for p in PROVIDERS if p in result}

    def delete_credential(self, user_id: int, provider: str) -> None:
        with self._connect() as db:
            was_default = db.execute(
                "SELECT is_default FROM llm_credentials WHERE user_id = ? AND provider = ?", (user_id, provider)
            ).fetchone()
            db.execute("DELETE FROM llm_credentials WHERE user_id = ? AND provider = ?", (user_id, provider))
            if was_default and was_default["is_default"]:
                other = db.execute("SELECT provider FROM llm_credentials WHERE user_id = ? LIMIT 1", (user_id,)).fetchone()
                if other:
                    db.execute(
                        "UPDATE llm_credentials SET is_default = 1 WHERE user_id = ? AND provider = ?",
                        (user_id, other["provider"]),
                    )

    def set_default(self, user_id: int, provider: str) -> None:
        with self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM llm_credentials WHERE user_id = ? AND provider = ?", (user_id, provider)
            ).fetchone():
                raise AccountError("Интеграция не настроена")
            db.execute("UPDATE llm_credentials SET is_default = (provider = ?) WHERE user_id = ?", (provider, user_id))

    def record_check(self, user_id: int, provider: str, ok: bool, message: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE llm_credentials SET last_check_ok = ?, last_check_message = ? WHERE user_id = ? AND provider = ?",
                (int(ok), message[:300], user_id, provider),
            )

    def default_provider(self, user_id: int) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT provider FROM llm_credentials WHERE user_id = ? ORDER BY is_default DESC LIMIT 1", (user_id,)
            ).fetchone()
        return row["provider"] if row else None

    def llm_config(self, user_id: int, provider: str | None = None) -> LLMConfig | None:
        """Полные настройки интеграции (с расшифрованным ключом) для обращения к модели."""
        provider = provider or self.default_provider(user_id)
        if provider is None:
            return None
        row = self._row(user_id, provider)
        if row is None:
            return None
        return LLMConfig(
            provider=provider,
            model=row["model"],
            api_key=self._decrypt(row["api_key_enc"]),
            base_url=row["base_url"],
            json_mode=bool(row["json_mode"]),
            temperature=0.2 if provider in ("anthropic", "local") else None,
        )

from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

def _default_app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "LoocieCoreAI"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", protected_namespaces=("settings_",))

    env: str = "dev"
    engine_name: str = "LoocieCoreAI Core Engine"
    engine_version: str = "3.0.0"

    loocie_internal_key: str | None = None
    loocie_api_key: str | None = None
    loocie_text_unlock: str | None = None

    model_provider: str = "ollama"
    model_endpoint: str = "http://127.0.0.1:11434"
    model_name: str = "phi3:mini"

    connect_timeout_seconds: float = 2.0
    read_timeout_seconds: float = 15.0
    memory_max_messages: int = 12
    warmup_on_start: bool = True

    app_support_dir: Path = _default_app_support_dir()
    logs_dirname: str = "logs"

    @property
    def effective_api_key(self) -> str | None:
        return self.loocie_internal_key or self.loocie_api_key

    @property
    def logs_dir(self) -> Path:
        return self.app_support_dir / self.logs_dirname

    def ensure_dirs(self) -> None:
        self.app_support_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def redact(self) -> dict:
        return {
            "env": self.env,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "model_provider": self.model_provider,
            "model_endpoint": self.model_endpoint,
            "model_name": self.model_name,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "memory_max_messages": self.memory_max_messages,
            "warmup_on_start": self.warmup_on_start,
            "app_support_dir": str(self.app_support_dir),
            "logs_dir": str(self.logs_dir),
            "has_api_key": bool(self.effective_api_key),
            "has_text_unlock": bool(self.loocie_text_unlock),
        }

settings = Settings()

_override_dir = os.getenv("LOOCIE_APP_SUPPORT_DIR")
if _override_dir:
    settings.app_support_dir = Path(_override_dir).expanduser().resolve()

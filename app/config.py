from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core runtime
    loocie_env: str = Field(default="dev")
    loocie_vault_path: str = Field(default="")

    # V3 identity (NEW START)
    loocie_api_title: str = Field(default="LoocieAI V3 Master")
    loocie_api_version: str = Field(default="3.0.0")

    # Security / debug
    loocie_debug: bool = Field(default=False)
    loocie_secret_key: str = Field(default="CHANGE-ME-IN-PRODUCTION")

    @field_validator("loocie_env")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"dev", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"LOOCIE_ENV must be one of {allowed}")
        return v.lower()

    @property
    def is_dev(self) -> bool:
        return self.loocie_env == "dev"

    @property
    def is_production(self) -> bool:
        return self.loocie_env == "production"

    @property
    def vault_is_configured(self) -> bool:
        return bool(self.loocie_vault_path and self.loocie_vault_path.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def invalidate_settings_cache() -> None:
    get_settings.cache_clear()

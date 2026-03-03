# =============================================================================

# app/config.py

# Loocie AI V2 Master — Configuration Engine

# IP: Seven Holy Creations, LLC | Operated by: iVenomLegacy Studios, LLC

# Standard: IVL-BVS-2026-03-02-001

# =============================================================================

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
“””
Central configuration object for the Loocie AI Engine.
All values are loaded from the .env file or environment variables.
“””

```
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",  # Ignore unknown env vars silently
)

# -------------------------------------------------------------------------
# Core Environment
# -------------------------------------------------------------------------
loocie_env: str = Field(
    default="dev",
    description="Runtime environment: dev | staging | production",
)

# -------------------------------------------------------------------------
# Business Vault Standard (BVS) — IVL-BVS-2026-03-02-001
# -------------------------------------------------------------------------
loocie_vault_path: str = Field(
    default="",
    description="Absolute path to the mounted Business Vault (external drive).",
)

# -------------------------------------------------------------------------
# API / Engine Identity
# -------------------------------------------------------------------------
loocie_api_title: str = Field(default="LoocieAI V2 Master")
loocie_api_version: str = Field(default="2.0.0")
loocie_debug: bool = Field(default=False)

# -------------------------------------------------------------------------
# Security
# -------------------------------------------------------------------------
loocie_secret_key: str = Field(
    default="CHANGE-ME-IN-PRODUCTION",
    description="Secret key for signing tokens and policy packs.",
)
loocie_allowed_hosts: list[str] = Field(
    default=["localhost", "127.0.0.1"],
    description="Comma-separated allowed hosts for Zero-Trust validation.",
)

# -------------------------------------------------------------------------
# Validators
# -------------------------------------------------------------------------
@field_validator("loocie_env")
@classmethod
def validate_environment(cls, v: str) -> str:
    allowed = {"dev", "staging", "production"}
    if v.lower() not in allowed:
        raise ValueError(f"LOOCIE_ENV must be one of {allowed}, got: '{v}'")
    return v.lower()

@field_validator("loocie_secret_key")
@classmethod
def warn_default_secret(cls, v: str) -> str:
    if v == "CHANGE-ME-IN-PRODUCTION":
        import warnings
        warnings.warn(
            "[LOOCIE SECURITY] Default secret key is in use. "
            "Set LOOCIE_SECRET_KEY in your .env before staging/production.",
            stacklevel=2,
        )
    return v

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
@property
def is_dev(self) -> bool:
    return self.loocie_env == "dev"

@property
def is_production(self) -> bool:
    return self.loocie_env == "production"

@property
def vault_is_configured(self) -> bool:
    return bool(self.loocie_vault_path and self.loocie_vault_path.strip())
```

# —————————————————————————–

# Singleton accessor — use get_settings() everywhere, never instantiate directly

# —————————————————————————–

@lru_cache(maxsize=1)
def get_settings() -> Settings:
“””
Returns a cached Settings singleton.
Call invalidate_settings_cache() in tests to reset.
“””
return Settings()

def invalidate_settings_cache() -> None:
“”“Clears the settings cache. Use in tests or after .env changes.”””
get_settings.cache_clear()
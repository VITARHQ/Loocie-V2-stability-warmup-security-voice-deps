# =============================================================================

# app/core/vault.py

# Loocie AI V2 Master — Business Vault Hard-Gate

# IP: Seven Holy Creations, LLC | Operated by: iVenomLegacy Studios, LLC

# Standard: IVL-BVS-2026-03-02-001

# =============================================================================

# 

# The Vault Hard-Gate enforces the Business Vault Standard (BVS).

# The Loocie Engine will REFUSE TO BOOT if the vault is not mounted

# and structurally valid.

# 

# Required Vault Structure:

# <VAULT_ROOT>/

# ├── 00_CONFIG/          ← Policy Packs injected here by Master Unit

# ├── 01_KNOWLEDGE_BASE/  ← Live knowledge + vector DB

# ├── 02_MEMORY_DB/       ← Agent memory / session state

# └── 03_LOGS_AUDIT/      ← Daily query logs + decision records

# =============================================================================

import os
import logging
from pathlib import Path
from app.config import get_settings

logger = logging.getLogger(“loocie.vault”)

# BVS-required folders (Drive A — Ops)

REQUIRED_VAULT_FOLDERS: list[str] = [
“00_CONFIG”,
“01_KNOWLEDGE_BASE”,
“02_MEMORY_DB”,
“03_LOGS_AUDIT”,
]

class VaultError(RuntimeError):
“”“Raised when the Business Vault fails validation. Engine will not boot.”””
pass

class VaultStatus:
“”“Snapshot of vault state returned after verification.”””

```
def __init__(self, vault_path: str, missing_folders: list[str]):
    self.vault_path = vault_path
    self.missing_folders = missing_folders
    self.is_valid = len(missing_folders) == 0

def __repr__(self) -> str:
    status = "VALID" if self.is_valid else f"INVALID (missing: {self.missing_folders})"
    return f"<VaultStatus path='{self.vault_path}' status={status}>"
```

def verify_vault(strict: bool = True) -> VaultStatus:
“””
Verifies the Business Vault is mounted and structurally valid.

```
Args:
    strict: If True (default), raises VaultError on failure.
            If False, returns VaultStatus without raising (use for health checks).

Returns:
    VaultStatus — snapshot of vault state.

Raises:
    VaultError — if strict=True and vault is missing or incomplete.
"""
settings = get_settings()
vault_path = settings.loocie_vault_path.strip()

# --- Gate 1: Vault path must be configured ---
if not vault_path:
    msg = (
        "[VAULT HARD-GATE] LOOCIE_VAULT_PATH is not set in .env. "
        "Mount the Business Vault and set the path before starting the engine."
    )
    logger.critical(msg)
    if strict:
        raise VaultError(msg)
    return VaultStatus(vault_path="<not set>", missing_folders=REQUIRED_VAULT_FOLDERS)

vault_root = Path(vault_path)

# --- Gate 2: Vault root directory must exist ---
if not vault_root.exists():
    msg = (
        f"[VAULT HARD-GATE] Vault path does not exist: '{vault_root}'. "
        "Ensure the Business Vault drive is mounted and the path is correct."
    )
    logger.critical(msg)
    if strict:
        raise VaultError(msg)
    return VaultStatus(vault_path=str(vault_root), missing_folders=REQUIRED_VAULT_FOLDERS)

if not vault_root.is_dir():
    msg = f"[VAULT HARD-GATE] Vault path exists but is not a directory: '{vault_root}'."
    logger.critical(msg)
    if strict:
        raise VaultError(msg)
    return VaultStatus(vault_path=str(vault_root), missing_folders=REQUIRED_VAULT_FOLDERS)

# --- Gate 3: All required BVS folders must be present ---
missing: list[str] = []
for folder in REQUIRED_VAULT_FOLDERS:
    folder_path = vault_root / folder
    if not folder_path.exists() or not folder_path.is_dir():
        missing.append(folder)

status = VaultStatus(vault_path=str(vault_root), missing_folders=missing)

if missing:
    msg = (
        f"[VAULT HARD-GATE] Vault structure is incomplete. "
        f"Missing BVS folders: {missing}. "
        f"Run vault_init() to create the required structure."
    )
    logger.critical(msg)
    if strict:
        raise VaultError(msg)
else:
    logger.info(f"[VAULT] Business Vault verified at: '{vault_root}'")

return status
```

def vault_init() -> VaultStatus:
“””
Creates the required BVS folder structure inside the configured vault path.
Safe to run multiple times (idempotent).

```
Returns:
    VaultStatus after initialization.

Raises:
    VaultError — if LOOCIE_VAULT_PATH is not configured.
"""
settings = get_settings()
vault_path = settings.loocie_vault_path.strip()

if not vault_path:
    raise VaultError(
        "[VAULT INIT] Cannot initialize vault — LOOCIE_VAULT_PATH is not set."
    )

vault_root = Path(vault_path)
vault_root.mkdir(parents=True, exist_ok=True)

for folder in REQUIRED_VAULT_FOLDERS:
    folder_path = vault_root / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VAULT INIT] Ensured folder: {folder_path}")

logger.info(f"[VAULT INIT] Business Vault initialized at: '{vault_root}'")
return verify_vault(strict=False)
```

def get_vault_path(subfolder: str = “”) -> Path:
“””
Returns the absolute path to a vault subfolder.

```
Args:
    subfolder: One of the BVS folder names (e.g. '00_CONFIG', '03_LOGS_AUDIT').

Returns:
    Path object.

Raises:
    VaultError — if vault is not configured.
"""
settings = get_settings()
vault_path = settings.loocie_vault_path.strip()

if not vault_path:
    raise VaultError("LOOCIE_VAULT_PATH is not configured.")

base = Path(vault_path)
return base / subfolder if subfolder else base
```
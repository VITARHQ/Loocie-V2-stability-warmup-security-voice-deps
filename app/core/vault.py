import logging
from pathlib import Path
from app.config import get_settings

logger = logging.getLogger("loocie.vault")

REQUIRED_VAULT_FOLDERS = [
    "00_CONFIG",
    "01_KNOWLEDGE_BASE",
    "02_MEMORY_DB",
    "03_LOGS_AUDIT",
]


class VaultError(RuntimeError):
    pass


class VaultStatus:
    def __init__(self, vault_path: str, missing_folders: list):
        self.vault_path = vault_path
        self.missing_folders = missing_folders
        self.is_valid = len(missing_folders) == 0

    def __repr__(self):
        status = "VALID" if self.is_valid else f"INVALID (missing: {self.missing_folders})"
        return f"<VaultStatus path={self.vault_path} status={status}>"


def verify_vault(strict: bool = True) -> VaultStatus:
    settings = get_settings()
    vault_path = settings.loocie_vault_path.strip()

    if not vault_path:
        msg = "[VAULT] LOOCIE_VAULT_PATH is not set in .env"
        logger.warning(msg)
        if strict:
            raise VaultError(msg)
        return VaultStatus(vault_path="<not set>", missing_folders=REQUIRED_VAULT_FOLDERS)

    vault_root = Path(vault_path)

    if not vault_root.exists() or not vault_root.is_dir():
        msg = f"[VAULT] Vault path does not exist: {vault_root}"
        logger.warning(msg)
        if strict:
            raise VaultError(msg)
        return VaultStatus(vault_path=str(vault_root), missing_folders=REQUIRED_VAULT_FOLDERS)

    missing = [f for f in REQUIRED_VAULT_FOLDERS if not (vault_root / f).is_dir()]
    status = VaultStatus(vault_path=str(vault_root), missing_folders=missing)

    if missing and strict:
        raise VaultError(f"[VAULT] Missing folders: {missing}")

    return status


def vault_init() -> VaultStatus:
    settings = get_settings()
    vault_path = settings.loocie_vault_path.strip()
    if not vault_path:
        raise VaultError("[VAULT] LOOCIE_VAULT_PATH is not set")
    vault_root = Path(vault_path)
    vault_root.mkdir(parents=True, exist_ok=True)
    for folder in REQUIRED_VAULT_FOLDERS:
        (vault_root / folder).mkdir(parents=True, exist_ok=True)
    return verify_vault(strict=False)


def get_vault_path(subfolder: str = "") -> Path:
    settings = get_settings()
    vault_path = settings.loocie_vault_path.strip()
    if not vault_path:
        raise VaultError("LOOCIE_VAULT_PATH is not configured")
    base = Path(vault_path)
    return base / subfolder if subfolder else base

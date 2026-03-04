from fastapi import APIRouter
from app.config import get_settings
from app.core.vault import verify_vault
from app.app.logger_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
def health_check():
    settings = get_settings()
    vault_status = verify_vault(strict=False)

    status = {
        "status": "ok",
        "engine": settings.loocie_api_title,
        "version": settings.loocie_api_version,
        "env": settings.loocie_env,
        "vault": {
            "mounted": vault_status.is_valid,
            "path": vault_status.vault_path,
            "missing_folders": vault_status.missing_folders,
        },
    }

    logger.debug("[HEALTH] Health check requested - vault_mounted=%s", vault_status.is_valid)
    return status

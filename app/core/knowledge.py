import os
from pathlib import Path
from app.config import get_settings
from app.app.logger_config import get_logger

logger = get_logger(__name__)


def load_knowledge_base() -> str:
    settings = get_settings()
    vault_path = settings.loocie_vault_path.strip()

    if not vault_path:
        logger.warning("[KB] No vault path set - skipping knowledge load")
        return ""

    kb_path = Path(vault_path) / "01_KNOWLEDGE_BASE"

    if not kb_path.exists():
        logger.warning("[KB] Knowledge base folder not found at %s", kb_path)
        return ""

    files = sorted(kb_path.glob("*.md")) + sorted(kb_path.glob("*.txt"))

    if not files:
        logger.info("[KB] No knowledge files found in vault")
        return ""

    knowledge = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
            knowledge.append(f"--- {file.name} ---\n{text}")
            logger.info("[KB] Loaded: %s", file.name)
        except Exception as e:
            logger.error("[KB] Failed to load %s: %s", file.name, str(e))

    combined = "\n\n".join(knowledge)
    logger.info("[KB] Knowledge base loaded - %d files, %d chars", len(files), len(combined))
    return combined

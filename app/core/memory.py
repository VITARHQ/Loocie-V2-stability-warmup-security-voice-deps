import json
from pathlib import Path
from datetime import datetime, timezone
from app.config import get_settings
from app.logging import get_logger

logger = get_logger(__name__)

MEMORY_FILE = "conversation_history.json"
MAX_HISTORY = 20


def get_memory_path() -> Path:
    settings = get_settings()
    return Path(settings.loocie_vault_path) / "02_MEMORY_DB" / MEMORY_FILE


def load_memory() -> list:
    path = get_memory_path()
    if not path.exists():
        logger.info("[MEMORY] No existing memory found - starting fresh")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("[MEMORY] Loaded %d messages from vault", len(data))
        return data[-MAX_HISTORY:]
    except Exception as e:
        logger.error("[MEMORY] Failed to load memory: %s", str(e))
        return []


def save_memory(history: list) -> None:
    path = get_memory_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        logger.debug("[MEMORY] Saved %d messages to vault", len(history))
    except Exception as e:
        logger.error("[MEMORY] Failed to save memory: %s", str(e))


def add_to_memory(history: list, role: str, content: str) -> list:
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return history[-MAX_HISTORY:]


def clear_memory() -> None:
    path = get_memory_path()
    if path.exists():
        path.unlink()
        logger.info("[MEMORY] Memory cleared")

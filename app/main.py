from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.config import get_settings
from app.logging import setup_logging, get_logger
from app.api.router import api_router
from app.core.vault import verify_vault
from app.core.llm import DEFAULT_MODEL  # use current default model for warmup

settings = get_settings()
setup_logging(level="DEBUG" if settings.is_dev else "INFO")
logger = get_logger(__name__)


async def warmup_ollama(model: str = DEFAULT_MODEL) -> None:
    """
    Auto-warmup: Send a tiny request to Ollama on startup so the first real user request
    isn't the cold-start hit.

    Uses short timeouts so it never blocks startup.
    """
    ollama_url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "warm up"}],
    }

    timeout = httpx.Timeout(connect=1.5, read=8.0, write=8.0, pool=8.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(ollama_url, json=payload)
            r.raise_for_status()
        logger.info("[WARMUP] Ollama warmup OK (model=%s)", model)
    except Exception as e:
        logger.warning("[WARMUP] Ollama warmup skipped/failed (model=%s): %s", model, str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[ENGINE] Loocie AI V2 starting up - env=%s", settings.loocie_env)

    if settings.is_production:
        verify_vault(strict=True)
        logger.info("[VAULT] Business Vault verified")
    else:
        status = verify_vault(strict=False)
        if not status.is_valid:
            logger.warning("[VAULT] Not mounted - vault features disabled in dev")
        else:
            logger.info("[VAULT] Business Vault verified at %s", status.vault_path)

    await warmup_ollama(model=DEFAULT_MODEL)

    logger.info("[ENGINE] Startup complete - ready to serve requests")
    yield
    logger.info("[ENGINE] Loocie AI shutting down")


app = FastAPI(
    title=settings.loocie_api_title,
    version=settings.loocie_api_version,
    debug=settings.loocie_debug,
    lifespan=lifespan,
)

app.include_router(api_router)
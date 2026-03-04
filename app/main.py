from contextlib import asynccontextmanager
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from app.config import get_settings
from app.logging import setup_logging, get_logger
from app.api.router import api_router
from app.core.vault import verify_vault
from app.core.llm import DEFAULT_MODEL

settings = get_settings()
setup_logging(level="DEBUG" if settings.is_dev else "INFO")
logger = get_logger(__name__)


def _get_api_key() -> str | None:
    key = os.getenv("LOOCIE_API_KEY")
    return key.strip() if key else None


async def warmup_ollama(model: str = DEFAULT_MODEL) -> None:
    ollama_url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "warm up"}],
    }

    timeout = httpx.Timeout(connect=1.5, read=15.0, write=10.0, pool=10.0)

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

    api_key = _get_api_key()
    if api_key:
        logger.info("[SECURITY] API key auth enabled (X-API-Key required; /health excluded)")
    else:
        logger.warning("[SECURITY] LOOCIE_API_KEY is not set. API key auth is DISABLED.")

    logger.info("[ENGINE] Startup complete - ready to serve requests")
    yield
    logger.info("[ENGINE] Loocie AI shutting down")


app = FastAPI(
    title=settings.loocie_api_title,
    version=settings.loocie_api_version,
    debug=settings.loocie_debug,
    lifespan=lifespan,
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # Allow health checks without auth
    if request.url.path == "/health":
        return await call_next(request)

    expected = _get_api_key()
    if not expected:
        # dev-friendly: if no key set, do not enforce
        return await call_next(request)

    provided = request.headers.get("X-API-Key", "")
    if provided != expected:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "message": "Missing or invalid X-API-Key"},
        )

    return await call_next(request)


app.include_router(api_router)
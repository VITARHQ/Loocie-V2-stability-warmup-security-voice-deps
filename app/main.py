
# =============================================================================

# app/main.py

# Loocie AI V2 Master — Engine Entry Point

# IP: Seven Holy Creations, LLC | Operated by: iVenomLegacy Studios, LLC

# =============================================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.router import api_router       # Central route registry
from app.config import get_settings
from app.core.vault import verify_vault, VaultError

# Load .env before anything else

load_dotenv()

# —————————————————————————–

# Lifespan: runs boot checks before the engine accepts traffic

# —————————————————————————–

@asynccontextmanager
async def lifespan(app: FastAPI):
“””
Startup: Verify Business Vault before accepting any requests.
Shutdown: Graceful teardown hooks go here.
“””
settings = get_settings()

```
# In production, vault MUST be mounted — hard gate
# In dev, warn but allow boot without vault for local development
if settings.is_production:
    verify_vault(strict=True)   # Raises VaultError → engine refuses to boot
else:
    status = verify_vault(strict=False)
    if not status.is_valid:
        import warnings
        warnings.warn(
            f"[LOOCIE DEV] Vault not mounted or incomplete: {status}. "
            "Set LOOCIE_VAULT_PATH in .env to test vault features.",
            stacklevel=2,
        )

yield  # Engine is live and serving requests

# --- Shutdown ---
# Add cleanup here (close DB connections, flush logs, etc.)
```

# —————————————————————————–

# FastAPI App Instance

# —————————————————————————–

settings = get_settings()

app = FastAPI(
title=settings.loocie_api_title,
version=settings.loocie_api_version,
debug=settings.loocie_debug,
lifespan=lifespan,
)

# Attach the central API router (routes live in app/api/routes/)

app.include_router(api_router)
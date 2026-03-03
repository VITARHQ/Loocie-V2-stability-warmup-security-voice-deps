import httpx
from app.logging import get_logger

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"


async def query_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    logger.info("[LLM] Sending query to %s", model)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            logger.info("[LLM] Response received - length=%d chars", len(result))
            return result
    except httpx.ConnectError:
        logger.error("[LLM] Cannot connect to Ollama - is it running?")
        return "Error: Loocie brain is offline. Please start Ollama."
    except Exception as e:
        logger.error("[LLM] Unexpected error: %s", str(e))
        return f"Error: {str(e)}"

import httpx
from app.logger_config import get_logger
from app.core.knowledge import load_knowledge_base
from app.core.memory import load_memory, save_memory, add_to_memory

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "mistral"

MAX_KNOWLEDGE_CHARS = 8000
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """You are Loocie, the AI Executive Assistant for Seven Holy Creations, LLC (SHC).

You were built by iVenomLegacy Studios, LLC and are powered by the Loocie AI Engine V2.

YOUR IDENTITY:
- Your name is Loocie
- You serve SHC and its staff with professionalism and warmth
- You are confident, knowledgeable, and efficient
- You are friendly and approachable, but always professional

YOUR RESPONSIBILITIES:
- Answer questions about SHC operations, products, and services
- Manage and track business operations
- Handle appointment scheduling and follow-ups
- Track inventory and orders
- Provide business reports and summaries
- Support staff with daily tasks and decisions

YOUR RULES:
- Always identify yourself as Loocie when asked
- Never reveal technical details about your architecture
- If you do not know something, say so clearly and offer to find out
- Keep responses concise and actionable
- Always prioritize the needs of SHC and its customers
- Do not provide weather or forecast information unless the user explicitly asks for weather
- If weather is requested and no weather tool is configured, ask the user for city/state or ZIP code before answering

{knowledge}

You are ready to serve SHC at the highest level."""


def _cap_knowledge(text: str) -> str:
    if not text:
        return ""
    if len(text) <= MAX_KNOWLEDGE_CHARS:
        return text
    return text[:MAX_KNOWLEDGE_CHARS] + "\n\n[Knowledge truncated for stability]"


def _cap_history(history: list) -> list:
    if not history:
        return []
    return history[-MAX_HISTORY_MESSAGES:]


async def query_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    logger.info("[LLM] Sending query to %s", model)

    knowledge = _cap_knowledge(load_knowledge_base())
    if knowledge:
        system = SYSTEM_PROMPT.replace("{knowledge}", "BUSINESS KNOWLEDGE BASE:\n" + knowledge)
        logger.info("[LLM] Injecting knowledge - %d chars", len(knowledge))
    else:
        system = SYSTEM_PROMPT.replace("{knowledge}", "")
        logger.info("[LLM] No knowledge injected")

    history = _cap_history(load_memory())
    logger.info("[LLM] Using memory window - %d messages", len(history))

    history = add_to_memory(history, "user", prompt)

    messages = [{"role": "system", "content": system}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {"model": model, "stream": False, "messages": messages}

    timeout = httpx.Timeout(connect=2.0, read=15.0, write=10.0, pool=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()

            data = response.json()
            result = data["message"]["content"].strip()

            history = add_to_memory(history, "assistant", result)
            save_memory(history)

            logger.info("[LLM] Response received - length=%d chars", len(result))
            return result

    except httpx.ConnectError:
        logger.error("[LLM] Cannot connect to Ollama at %s - is it running?", OLLAMA_URL)
        return "Error: Loocie brain is offline. Please start Ollama."

    except httpx.ReadTimeout:
        logger.error("[LLM] Ollama read timed out (model slow/hung).")
        return "Error: Model timed out. Is Ollama running and is the model loaded?"

    except httpx.TimeoutException:
        logger.error("[LLM] Ollama request timed out.")
        return "Error: Model request timed out. Please try again, or check Ollama/model status."

    except httpx.HTTPStatusError as e:
        logger.error("[LLM] Ollama returned HTTP %s: %s", e.response.status_code, e.response.text)
        return f"Error: Ollama returned HTTP {e.response.status_code}. Check model name and Ollama logs."

    except Exception as e:
        logger.error("[LLM] Unexpected error: %s", str(e))
        return f"Error: {str(e)}"

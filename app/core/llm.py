import httpx
from app.app.logger_config import get_logger
from app.core.knowledge import load_knowledge_base
from app.core.memory import load_memory, save_memory, add_to_memory

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "mistral"

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

{knowledge}

You are ready to serve SHC at the highest level."""


async def query_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    logger.info("[LLM] Sending query to %s", model)

    knowledge = load_knowledge_base()
    if knowledge:
        system = SYSTEM_PROMPT.replace("{knowledge}", "BUSINESS KNOWLEDGE BASE:\n" + knowledge)
    else:
        system = SYSTEM_PROMPT.replace("{knowledge}", "")

    history = load_memory()
    history = add_to_memory(history, "user", prompt)

    messages = [{"role": "system", "content": system}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            history = add_to_memory(history, "assistant", result)
            save_memory(history)
            logger.info("[LLM] Response received - length=%d chars", len(result))
            return result
    except httpx.ConnectError:
        logger.error("[LLM] Cannot connect to Ollama - is it running?")
        return "Error: Loocie brain is offline. Please start Ollama."
    except Exception as e:
        logger.error("[LLM] Unexpected error: %s", str(e))
        return f"Error: {str(e)}"

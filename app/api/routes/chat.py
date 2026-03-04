from fastapi import APIRouter
from pydantic import BaseModel

from app.core.llm import query_llm, DEFAULT_MODEL
from app.logger_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str
    # B) Default to the LLM module's DEFAULT_MODEL (phi3:mini)
    model: str = DEFAULT_MODEL


class ChatResponse(BaseModel):
    reply: str
    model: str


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("[CHAT] Received message - length=%d chars", len(request.message))
    try:
        reply = await query_llm(prompt=request.message, model=request.model)
        return ChatResponse(reply=reply, model=request.model)
    except Exception as e:
        logger.exception("[CHAT] query_llm failed")
        return ChatResponse(reply=f"Error: {e}", model=request.model)
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.llm import LLMError
from app.services.rag import chat
from app.utils.logger import logger

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={"error": "Request body must be a JSON object"})

    if "message" not in body or not str(body.get("message", "")).strip():
        raise HTTPException(status_code=400, detail={"error": "Message field is required"})

    if "sessionId" not in body or not str(body.get("sessionId", "")).strip():
        raise HTTPException(status_code=400, detail={"error": "sessionId field is required"})

    try:
        payload = ChatRequest(**body)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = first.get("loc", ["field"])[-1]
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid value for {field}"},
        )

    logger.info("Chat | session=%s | message_len=%d", payload.sessionId, len(payload.message))

    try:
        result = chat(payload.sessionId, payload.message.strip())
        return ChatResponse(**result)
    except LLMError as exc:
        msg = str(exc)
        if "API key" in msg or "Invalid" in msg:
            raise HTTPException(status_code=401, detail={"error": msg})
        if "Rate limit" in msg:
            raise HTTPException(status_code=429, detail={"error": msg})
        if "timed out" in msg.lower() or "connect" in msg.lower():
            raise HTTPException(status_code=503, detail={"error": msg})
        raise HTTPException(status_code=500, detail={"error": msg})
    except Exception as exc:
        logger.exception("Unexpected chat error")
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})

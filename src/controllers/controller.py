import logging

from fastapi import HTTPException
from pydantic import ValidationError

from src.controllers.validation import ChatResponse, validate_chat_message
from src.services.chat import chat

logger = logging.getLogger(__name__)
_UPSTREAM_FAILURE = "The legal assistant could not answer right now."


def handle_chat(body: object) -> ChatResponse:
    """Validate the request, ask the chat service, and return the reply."""
    try:
        request = validate_chat_message(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    try:
        reply = chat(request.message)
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=_UPSTREAM_FAILURE) from exc
    return ChatResponse(reply=reply)

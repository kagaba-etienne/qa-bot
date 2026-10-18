from fastapi import APIRouter, Body

from src.controllers.controller import handle_chat
from src.controllers.validation import ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: dict = Body(...)) -> ChatResponse:
    """Accept a user message and return the assistant reply."""
    return handle_chat(body)

from pydantic import BaseModel, Field, ValidationError, field_validator

MAX_MESSAGE_LENGTH = 4000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    reply: str


def validate_chat_message(body: object) -> ChatRequest:
    """Validate the chat JSON body. Raises ValidationError when it is invalid."""
    return ChatRequest.model_validate(body)


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ValidationError",
    "validate_chat_message",
]

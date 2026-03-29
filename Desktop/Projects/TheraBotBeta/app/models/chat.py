from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Cost tracking — None until the LLM responds (user messages have no cost)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class ChatRequest(BaseModel):
    user_id: UUID
    session_id: UUID | None = None  # None → create a new session
    content: str
    profile: Literal["default", "cheap"] = "cheap"

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class ChatResponse(BaseModel):
    session_id: UUID
    message: ChatMessage
    turn_count: int | None = None
    experiment_type: str | None = None  # set when this turn was an A/B experiment


# CompareVariant and CompareResponse are defined before StreamChunk so that
# StreamChunk can reference CompareResponse without a forward reference.

class CompareVariant(BaseModel):
    response: str
    model: str
    prompt_variant: str


class CompareResponse(BaseModel):
    session_id: UUID
    message_id: UUID
    experiment_type: str
    variant_a: CompareVariant
    variant_b: CompareVariant


class VoteRequest(BaseModel):
    session_id: UUID
    message_id: UUID
    experiment_type: str
    variant_a: str  # prompt_variant identifier from the original response
    variant_b: str  # prompt_variant identifier from the original response
    winner: Literal["a", "b"]


class StreamChunkUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class StreamChunk(BaseModel):
    session_id: UUID
    message_id: UUID
    delta: str
    done: bool = False
    turn_count: int | None = None          # present on done chunks
    usage: StreamChunkUsage | None = None  # present on done chunks (normal turns)
    experiment: CompareResponse | None = None  # present on done chunks (experiment turns)

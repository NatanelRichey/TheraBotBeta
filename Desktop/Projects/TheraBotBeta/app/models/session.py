from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.chat import ChatMessage


class SessionCreate(BaseModel):
    user_id: UUID


class Session(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionSummary(BaseModel):
    id: UUID
    user_id: UUID
    message_count: int
    created_at: datetime
    updated_at: datetime

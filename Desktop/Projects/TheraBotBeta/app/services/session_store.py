# In-memory session store for Phase 1 (local dev + demos).
#
# SWAP POINT: Replace this class with a Redis or DB-backed implementation
# that exposes the same async interface. Business logic in ChatService
# never touches storage directly — only this interface.

import asyncio
from uuid import UUID, uuid4

from app.models.chat import ChatMessage
from app.models.session import Session, SessionSummary


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[UUID, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, user_id: UUID, session_id: UUID | None = None) -> Session:
        async with self._lock:
            session = Session(id=session_id or uuid4(), user_id=user_id)
            self._sessions[session.id] = session
            return session

    async def get(self, session_id: UUID) -> Session | None:
        return self._sessions.get(session_id)

    async def append_message(self, session_id: UUID, message: ChatMessage) -> Session:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            session.messages.append(message)
            return session

    async def list_for_user(self, user_id: UUID) -> list[SessionSummary]:
        return [
            SessionSummary(
                id=s.id,
                user_id=s.user_id,
                message_count=len(s.messages),
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in self._sessions.values()
            if s.user_id == user_id
        ]

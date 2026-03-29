from functools import lru_cache

from app.services.chat import ChatService
from app.services.llm.router import get_cheap_router, get_router
from app.services.session_store import SessionStore


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


def get_chat_service() -> ChatService:
    return ChatService(
        default_router=get_router(),
        cheap_router=get_cheap_router(),
        store=get_session_store(),
    )

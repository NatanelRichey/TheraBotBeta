from functools import lru_cache

from app.services.chat import ChatService
from app.services.llm.router import get_cheap_router, get_router
from app.services.prompts.experiments import ExperimentRunner
from app.services.rag.retriever import KnowledgeRetriever, get_retriever
from app.services.session_store import SessionStore


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


@lru_cache
def get_experiment_runner() -> ExperimentRunner:
    return ExperimentRunner(
        default_router=get_router(),
        cheap_router=get_cheap_router(),
    )


@lru_cache
def get_knowledge_retriever() -> KnowledgeRetriever:
    return get_retriever()


def get_chat_service() -> ChatService:
    return ChatService(
        default_router=get_router(),
        cheap_router=get_cheap_router(),
        store=get_session_store(),
        experiment_runner=get_experiment_runner(),
        retriever=get_knowledge_retriever(),
    )

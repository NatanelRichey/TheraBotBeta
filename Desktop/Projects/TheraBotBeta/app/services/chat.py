from collections.abc import AsyncIterator
from uuid import uuid4

from app.core.logging import get_logger
from app.models.chat import ChatMessage, ChatRequest, ChatResponse, MessageRole, StreamChunk, StreamChunkUsage
from app.models.session import Session
from app.services.llm.router import LLMRouter
from app.services.session_store import SessionStore

logger = get_logger(__name__)

# TODO: Replace with prompt pipeline (Phase 2)
SYSTEM_PROMPT = """You are a supportive wellness companion.
You listen with empathy, ask thoughtful questions, and help users
reflect on their mental and emotional wellbeing. You are not a
licensed therapist and do not provide clinical advice."""


class ChatService:
    def __init__(
        self,
        default_router: LLMRouter,
        cheap_router: LLMRouter,
        store: SessionStore,
    ) -> None:
        self._default_router = default_router
        self._cheap_router = cheap_router
        self._store = store

    def _get_router(self, request: ChatRequest) -> LLMRouter:
        return self._cheap_router if request.profile == "cheap" else self._default_router

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session = await self._resolve_session(request)
        user_message = ChatMessage(role=MessageRole.user, content=request.content)
        await self._store.append_message(session.id, user_message)

        llm_messages = self._build_llm_messages(session, user_message)
        response = await self._get_router(request).complete(llm_messages)

        assistant_message = ChatMessage(
            role=MessageRole.assistant,
            content=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )
        await self._store.append_message(session.id, assistant_message)

        logger.info(
            "chat_complete",
            session_id=str(session.id),
            user_id=str(request.user_id),
            model=response.model,
            cost_usd=response.cost_usd,
        )

        return ChatResponse(session_id=session.id, message=assistant_message)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        session = await self._resolve_session(request)
        user_message = ChatMessage(role=MessageRole.user, content=request.content)
        await self._store.append_message(session.id, user_message)

        llm_messages = self._build_llm_messages(session, user_message)
        message_id = uuid4()
        accumulated: list[str] = []

        async for chunk in self._get_router(request).stream(llm_messages):
            if chunk.done:
                if chunk.delta:
                    accumulated.append(chunk.delta)
                assistant_message = ChatMessage(
                    id=message_id,
                    role=MessageRole.assistant,
                    content="".join(accumulated),
                    model=chunk.model,
                    input_tokens=chunk.input_tokens,
                    output_tokens=chunk.output_tokens,
                    cost_usd=chunk.cost_usd,
                )
                await self._store.append_message(session.id, assistant_message)

                logger.info(
                    "stream_complete",
                    session_id=str(session.id),
                    user_id=str(request.user_id),
                    model=chunk.model,
                    cost_usd=chunk.cost_usd,
                )

                input_tok = chunk.input_tokens or 0
                output_tok = chunk.output_tokens or 0
                yield StreamChunk(
                    session_id=session.id,
                    message_id=message_id,
                    delta="",
                    done=True,
                    usage=StreamChunkUsage(
                        total_tokens=input_tok + output_tok,
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        cost_usd=chunk.cost_usd or 0.0,
                    ),
                )
            else:
                accumulated.append(chunk.delta)
                yield StreamChunk(
                    session_id=session.id,
                    message_id=message_id,
                    delta=chunk.delta,
                    done=False,
                )

    async def _resolve_session(self, request: ChatRequest) -> Session:
        if request.session_id is not None:
            session = await self._store.get(request.session_id)
            if session is not None:
                return session
            # First message for this session_id — create and store under the same ID
            return await self._store.create(user_id=request.user_id, session_id=request.session_id)
        return await self._store.create(user_id=request.user_id)

    def _build_llm_messages(
        self, session: Session, new_message: ChatMessage
    ) -> list[ChatMessage]:
        system = ChatMessage(role=MessageRole.system, content=SYSTEM_PROMPT)
        # session.messages already includes new_message (appended before this call)
        return [system, *session.messages]

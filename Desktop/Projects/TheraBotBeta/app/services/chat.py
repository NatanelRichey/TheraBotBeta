from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import uuid4

from app.core.logging import get_logger
from app.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompareResponse,
    CompareVariant,
    MessageRole,
    StreamChunk,
    StreamChunkUsage,
    VoteRequest,
)
from app.models.session import Session
from app.services.llm.router import LLMRouter
from app.services.prompts.experiments import ExperimentRunner
from app.services.prompts.pipeline import assemble
from app.services.session_store import SessionStore

logger = get_logger(__name__)

_EXPERIMENT_INTERVAL = 5  # run an A/B experiment every Nth user turn


class ChatService:
    def __init__(
        self,
        default_router: LLMRouter,
        cheap_router: LLMRouter,
        store: SessionStore,
        experiment_runner: ExperimentRunner,
    ) -> None:
        self._default_router = default_router
        self._cheap_router = cheap_router
        self._store = store
        self._experiment_runner = experiment_runner

    def _get_router(self, request: ChatRequest) -> LLMRouter:
        return self._cheap_router if request.profile == "cheap" else self._default_router

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session = await self._resolve_session(request)
        user_message = ChatMessage(role=MessageRole.user, content=request.content)
        await self._store.append_message(session.id, user_message)
        session.turn_count += 1

        if session.turn_count % _EXPERIMENT_INTERVAL == 0:
            compare_result = await self._run_experiment(session, request)
            # Surface experiment metadata so the caller knows this was a compare turn.
            return ChatResponse(
                session_id=session.id,
                message=ChatMessage(
                    id=compare_result.message_id,
                    role=MessageRole.assistant,
                    content=compare_result.variant_a.response,
                    model=compare_result.variant_a.model,
                ),
                turn_count=session.turn_count,
                experiment_type=compare_result.experiment_type,
            )

        system_prompt = assemble()
        llm_messages = self._build_llm_messages(session, system_prompt)
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
            model=assistant_message.model,
            turn_count=session.turn_count,
        )

        return ChatResponse(
            session_id=session.id,
            message=assistant_message,
            turn_count=session.turn_count,
        )

    async def compare(self, request: ChatRequest) -> CompareResponse:
        """Explicit A/B compare — always runs an experiment regardless of turn count."""
        session = await self._resolve_session(request)
        user_message = ChatMessage(role=MessageRole.user, content=request.content)
        await self._store.append_message(session.id, user_message)
        session.turn_count += 1
        return await self._run_experiment(session, request)

    async def record_vote(self, vote: VoteRequest) -> None:
        """Log a comparison vote. No storage needed — the log is the record."""
        logger.info(
            "comparison_vote",
            session_id=str(vote.session_id),
            message_id=str(vote.message_id),
            experiment_type=vote.experiment_type,
            variant_a=vote.variant_a,
            variant_b=vote.variant_b,
            winner=vote.winner,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        session = await self._resolve_session(request)
        user_message = ChatMessage(role=MessageRole.user, content=request.content)
        await self._store.append_message(session.id, user_message)
        session.turn_count += 1

        if session.turn_count % _EXPERIMENT_INTERVAL == 0:
            compare_result = await self._run_experiment(session, request)
            yield StreamChunk(
                session_id=session.id,
                message_id=compare_result.message_id,
                delta="",
                done=True,
                turn_count=session.turn_count,
                experiment=compare_result,
            )
            return

        system_prompt = assemble()
        llm_messages = self._build_llm_messages(session, system_prompt)
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
                    turn_count=session.turn_count,
                )

                input_tok = chunk.input_tokens or 0
                output_tok = chunk.output_tokens or 0
                yield StreamChunk(
                    session_id=session.id,
                    message_id=message_id,
                    delta="",
                    done=True,
                    turn_count=session.turn_count,
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

    async def _run_experiment(self, session: Session, request: ChatRequest) -> CompareResponse:
        """Run A/B experiment. Assumes user message is already appended to session."""
        message_id = uuid4()
        exp_result = await self._experiment_runner.run(
            session.messages, session.id, router=self._get_router(request)
        )

        # Variant A is the canonical response saved to session history.
        assistant_message = ChatMessage(
            id=message_id,
            role=MessageRole.assistant,
            content=exp_result.variant_a.response,
            model=exp_result.variant_a.model,
        )
        await self._store.append_message(session.id, assistant_message)

        logger.info(
            "experiment_complete",
            session_id=str(session.id),
            experiment_type=exp_result.experiment_type.value,
            turn_count=session.turn_count,
        )

        return CompareResponse(
            session_id=session.id,
            message_id=message_id,
            experiment_type=exp_result.experiment_type.value,
            variant_a=CompareVariant(
                response=exp_result.variant_a.response,
                model=exp_result.variant_a.model,
                prompt_variant=exp_result.variant_a.prompt_variant,
            ),
            variant_b=CompareVariant(
                response=exp_result.variant_b.response,
                model=exp_result.variant_b.model,
                prompt_variant=exp_result.variant_b.prompt_variant,
            ),
        )

    async def _resolve_session(self, request: ChatRequest) -> Session:
        if request.session_id is not None:
            session = await self._store.get(request.session_id)
            if session is not None:
                return session
            # First message for this session_id — create and store under the same ID
            return await self._store.create(user_id=request.user_id, session_id=request.session_id)
        return await self._store.create(user_id=request.user_id)

    def _build_llm_messages(self, session: Session, system_prompt: str) -> list[ChatMessage]:
        system = ChatMessage(role=MessageRole.system, content=system_prompt)
        # session.messages already includes the new user message (appended before this call)
        return [system, *session.messages]

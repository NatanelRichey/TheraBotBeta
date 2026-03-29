from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, MessageRole
from app.services.llm.base import BaseLLMProvider, LLMResponse, LLMStreamChunk
from app.services.llm.cost_tracker import calculate_cost

logger = get_logger(__name__)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries handled by router via tenacity
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        logger.debug(
            "openai_complete",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        input_tokens = 0
        output_tokens = 0
        last_delta = ""

        stream = await self._client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        # Yield content deltas until the first finish_reason="stop".
        # Do NOT yield done=True here — some providers (e.g. OpenRouter) send a
        # second chunk with finish_reason="stop" that carries the usage data, which
        # would cause the done signal to fire twice.
        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            finish_reason = chunk.choices[0].finish_reason

            if finish_reason == "stop":
                last_delta = delta  # capture final token if provider includes it here
                break

            yield LLMStreamChunk(delta=delta)

        # Drain remaining chunks — usage data may arrive after the stop signal.
        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        cost = calculate_cost(model, input_tokens, output_tokens)
        logger.debug(
            "openai_stream_complete",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        yield LLMStreamChunk(
            delta=last_delta,
            done=True,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

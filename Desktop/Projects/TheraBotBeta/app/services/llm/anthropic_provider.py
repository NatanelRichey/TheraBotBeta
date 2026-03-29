from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, MessageRole
from app.services.llm.base import BaseLLMProvider, LLMResponse, LLMStreamChunk
from app.services.llm.cost_tracker import calculate_cost

logger = get_logger(__name__)

# Anthropic requires system messages to be passed separately, not in the
# messages list. This function splits them out.
def _split_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict]]:
    system_parts = [m.content for m in messages if m.role == MessageRole.system]
    conversation = [
        {"role": m.role.value, "content": m.content}
        for m in messages
        if m.role != MessageRole.system
    ]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conversation


class AnthropicProvider(BaseLLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries handled by router via tenacity
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> LLMResponse:
        system, conversation = _split_messages(messages)

        response = await self._client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 2048),
            system=system or anthropic.NOT_GIVEN,
            messages=conversation,
            **kwargs,
        )

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        logger.debug(
            "anthropic_complete",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        return LLMResponse(
            content=content,
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
        system, conversation = _split_messages(messages)
        input_tokens = 0
        output_tokens = 0

        async with self._client.messages.stream(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 2048),
            system=system or anthropic.NOT_GIVEN,
            messages=conversation,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield LLMStreamChunk(delta=text)

            # Final message has full usage
            final = await stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
            cost = calculate_cost(model, input_tokens, output_tokens)

            logger.debug(
                "anthropic_stream_complete",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

            yield LLMStreamChunk(
                delta="",
                done=True,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

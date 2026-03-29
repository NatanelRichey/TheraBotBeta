from collections.abc import AsyncIterator
from functools import lru_cache

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import BudgetExceededError, LLMProviderError
from app.core.logging import get_logger
from app.models.chat import ChatMessage
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import BaseLLMProvider, LLMResponse, LLMStreamChunk
from app.services.llm.cost_tracker import CostTracker
from app.services.llm.openai_provider import OpenAIProvider

logger = get_logger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _make_openrouter_provider() -> OpenAIProvider:
    settings = get_settings()
    return OpenAIProvider(
        api_key=settings.openrouter_api_key,
        base_url=_OPENROUTER_BASE_URL,
    )


def _make_retry(max_retries: int):
    return retry(
        retry=retry_if_exception_type(LLMProviderError),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )


class LLMRouter:
    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
        cost_tracker: CostTracker,
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cost_tracker = cost_tracker
        settings = get_settings()
        self._primary_model = primary_model or settings.primary_model
        self._fallback_model = fallback_model or settings.fallback_model
        self._max_retries = settings.llm_max_retries

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        # BudgetExceededError is never retried — it propagates immediately
        self._cost_tracker.record  # touch to ensure tracker is alive

        chosen_model = model or self._primary_model

        try:
            response = await self._call_with_retry(
                self._primary, chosen_model, messages, **kwargs
            )
            self._cost_tracker.record(chosen_model, response.input_tokens, response.output_tokens)
            return response
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning(
                "llm_primary_failed_falling_back",
                primary_model=chosen_model,
                error=str(exc),
            )

        response = await self._call_with_retry(
            self._fallback, self._fallback_model, messages, **kwargs
        )
        self._cost_tracker.record(self._fallback_model, response.input_tokens, response.output_tokens)
        return response

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        chosen_model = model or self._primary_model

        try:
            async for chunk in self._primary.stream(messages, chosen_model, **kwargs):
                if chunk.done and chunk.input_tokens is not None:
                    self._cost_tracker.record(
                        chosen_model, chunk.input_tokens, chunk.output_tokens or 0
                    )
                yield chunk
            return
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning(
                "llm_primary_stream_failed_falling_back",
                primary_model=chosen_model,
                error=str(exc),
            )

        async for chunk in self._fallback.stream(messages, self._fallback_model, **kwargs):
            if chunk.done and chunk.input_tokens is not None:
                self._cost_tracker.record(
                    self._fallback_model, chunk.input_tokens, chunk.output_tokens or 0
                )
            yield chunk

    async def _call_with_retry(
        self,
        provider: BaseLLMProvider,
        model: str,
        messages: list[ChatMessage],
        **kwargs,
    ) -> LLMResponse:
        @_make_retry(self._max_retries)
        async def _call() -> LLMResponse:
            try:
                return await provider.complete(messages, model, **kwargs)
            except BudgetExceededError:
                raise
            except Exception as exc:
                raise LLMProviderError(str(exc)) from exc

        return await _call()


@lru_cache
def get_router() -> LLMRouter:
    settings = get_settings()
    if settings.use_openrouter:
        provider = _make_openrouter_provider()
        return LLMRouter(
            primary=provider,
            fallback=provider,
            cost_tracker=CostTracker(),
            primary_model=settings.openrouter_default_primary_model,
            fallback_model=settings.openrouter_default_fallback_model,
        )
    return LLMRouter(
        primary=OpenAIProvider(),
        fallback=AnthropicProvider(),
        cost_tracker=CostTracker(),
    )


@lru_cache
def get_cheap_router() -> LLMRouter:
    settings = get_settings()
    if settings.use_openrouter:
        provider = _make_openrouter_provider()
        return LLMRouter(
            primary=provider,
            fallback=provider,
            cost_tracker=CostTracker(),
            primary_model=settings.openrouter_cheap_primary_model,
            fallback_model=settings.openrouter_cheap_fallback_model,
        )
    return LLMRouter(
        primary=OpenAIProvider(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        ),
        fallback=OpenAIProvider(
            api_key=settings.kimi_api_key,
            base_url="https://api.moonshot.cn/v1",
        ),
        cost_tracker=CostTracker(),
        primary_model=settings.deepseek_model,
        fallback_model=settings.kimi_model,
    )

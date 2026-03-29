from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.models.chat import ChatMessage


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class LLMStreamChunk:
    delta: str
    done: bool = False
    # Only populated on the final chunk (done=True)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]: ...

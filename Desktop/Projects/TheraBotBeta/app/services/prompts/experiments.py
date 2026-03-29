"""A/B experiment framework for prompt and model comparison.

Every 10th conversation turn (tracked via Session.turn_count), the chat service
calls ExperimentRunner.run() instead of the normal single-LLM flow. Both
variants are fired concurrently via asyncio.gather, and variant_a is saved to
the session as the canonical response.

The POST /chat/compare endpoint exposes the same runner on-demand, returning
both variants to the caller for side-by-side UI rendering.
"""
import asyncio
import random
from dataclasses import dataclass, replace
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.chat import ChatMessage, MessageRole
from app.services.llm.router import LLMRouter
from app.services.prompts.pipeline import assemble

logger = get_logger(__name__)


class ExperimentType(str, Enum):
    """The four A/B experiment dimensions."""

    IDENTITY_TONE = "IDENTITY_TONE"
    FORMAT_LENGTH = "FORMAT_LENGTH"
    MODEL = "MODEL"
    IDENTITY_VERSION = "IDENTITY_VERSION"


class VariantResult(BaseModel):
    """Output from a single experiment variant."""

    response: str
    model: str
    prompt_variant: str


class ExperimentResult(BaseModel):
    """Full output from a two-variant experiment run."""

    experiment_type: ExperimentType
    variant_a: VariantResult
    variant_b: VariantResult


@dataclass
class _VariantConfig:
    """Internal configuration for one experiment variant."""

    router: LLMRouter
    identity_template: str = "identity_warm"
    identity_version: str = "v1"
    format_template: str = "format_short"
    format_version: str = "v1"
    model: str | None = None
    prompt_variant: str = "default"


class ExperimentRunner:
    """Fires two prompt/model variants concurrently and returns both results."""

    def __init__(self, default_router: LLMRouter, cheap_router: LLMRouter) -> None:
        self._default_router = default_router
        self._cheap_router = cheap_router

    async def run(
        self,
        messages: list[ChatMessage],
        session_id: UUID,
        experiment_type: ExperimentType | None = None,
        router: LLMRouter | None = None,
    ) -> ExperimentResult:
        """Run a comparison experiment with two variants concurrently.

        Args:
            messages: Conversation history *without* a system message.
                The runner prepends its own system message per variant.
            session_id: Used for structured log correlation only.
            experiment_type: Pin a specific type; randomly chosen if ``None``.

        Returns:
            :class:`ExperimentResult` containing both variant outputs.
        """
        exp_type = experiment_type or random.choice(list(ExperimentType))
        active_router = router or self._default_router
        config_a, config_b = self._get_variant_configs(exp_type, active_router)

        logger.info(
            "experiment_triggered",
            session_id=str(session_id),
            experiment_type=exp_type.value,
            variant_a=config_a.prompt_variant,
            variant_b=config_b.prompt_variant,
        )

        result_a, result_b = await asyncio.gather(
            self._run_variant(messages, config_a),
            self._run_variant(messages, config_b),
        )

        return ExperimentResult(
            experiment_type=exp_type,
            variant_a=result_a,
            variant_b=result_b,
        )

    async def _run_variant(
        self,
        messages: list[ChatMessage],
        config: _VariantConfig,
    ) -> VariantResult:
        """Build system prompt for this config, call LLM, return result."""
        system_prompt = assemble(
            identity_template=config.identity_template,
            identity_version=config.identity_version,
            format_template=config.format_template,
            format_version=config.format_version,
        )
        full_messages = [
            ChatMessage(role=MessageRole.system, content=system_prompt),
            *messages,
        ]
        response = await config.router.complete(full_messages, model=config.model)
        return VariantResult(
            response=response.content,
            model=response.model,
            prompt_variant=config.prompt_variant,
        )

    def _get_variant_configs(
        self, exp_type: ExperimentType, active_router: LLMRouter
    ) -> tuple[_VariantConfig, _VariantConfig]:
        """Return (config_a, config_b) for the given experiment type.

        For prompt/format/version experiments, both variants use ``active_router``
        so the profile selected by the user is respected. For MODEL experiments,
        the two routers are always compared against each other regardless of profile.
        """
        base = _VariantConfig(router=active_router)

        if exp_type == ExperimentType.IDENTITY_TONE:
            return (
                replace(base, prompt_variant="identity_warm_v1"),
                replace(base, identity_template="identity_clinical", prompt_variant="identity_clinical_v1"),
            )
        if exp_type == ExperimentType.FORMAT_LENGTH:
            return (
                replace(base, prompt_variant="format_short_v1"),
                replace(base, format_template="format_long", prompt_variant="format_long_v1"),
            )
        if exp_type == ExperimentType.MODEL:
            return (
                replace(base, router=self._default_router, prompt_variant="model_default"),
                replace(base, router=self._cheap_router, prompt_variant="model_cheap"),
            )
        # IDENTITY_VERSION
        return (
            replace(base, prompt_variant="identity_warm_v1"),
            replace(base, identity_version="v2", prompt_variant="identity_warm_v2"),
        )

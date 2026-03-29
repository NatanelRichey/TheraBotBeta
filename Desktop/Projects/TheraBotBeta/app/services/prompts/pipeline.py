"""Multi-stage system prompt assembly pipeline.

Stages (in order):
    1. Identity  — always included  (default: identity_warm_v1)
    2. Memory    — TODO: inject MemoryAgent summary (Phase 4)
    3. Safety    — injected only when escalation=True
    4. Format    — always included  (default: format_short_v1)

Each stage is a rendered template block separated by a horizontal rule so the
model can clearly distinguish the sections.
"""
from app.core.logging import get_logger
from app.services.prompts.templates import render

logger = get_logger(__name__)

_STAGE_SEPARATOR = "\n\n---\n\n"


def assemble(
    *,
    escalation: bool = False,
    knowledge_context: str = "",
    identity_template: str = "identity_warm",
    identity_version: str = "v1",
    format_template: str = "format_short",
    format_version: str = "v1",
    variables: dict[str, str] | None = None,
) -> str:
    """Assemble the final system prompt from ordered pipeline stages.

    Args:
        escalation: When ``True``, injects the safety escalation block
            (stage 3) between identity and format.
        knowledge_context: Pre-formatted RAG context block from
            ``grounding.format_context()``. Injected as stage 2 when non-empty.
        identity_template: Template name for the identity stage.
        identity_version: Version string for the identity template.
        format_template: Template name for the format stage.
        format_version: Version string for the format template.
        variables: Optional ``{key}`` substitution variables applied to every
            stage template.

    Returns:
        A single assembled system prompt string.
    """
    vars_ = variables or {}
    stages: list[str] = []

    # Stage 1: Identity — always present
    stages.append(render(identity_template, identity_version, **vars_))

    # Stage 2: Knowledge context — retrieved DBT chunks (Phase 3)
    if knowledge_context:
        stages.append(knowledge_context)

    # Stage 2.5: Memory — TODO: inject MemoryAgent summary (Phase 4)

    # Stage 3: Safety — only on escalation
    if escalation:
        stages.append(render("safety_escalation", "v1", **vars_))

    # Stage 4: Format — always present
    stages.append(render(format_template, format_version, **vars_))

    logger.debug(
        "pipeline_assembled",
        identity=f"{identity_template}_{identity_version}",
        format=f"{format_template}_{format_version}",
        escalation=escalation,
        has_knowledge_context=bool(knowledge_context),
        stage_count=len(stages),
    )

    return _STAGE_SEPARATOR.join(stages)

"""Format retrieved knowledge chunks into a system prompt context block."""

from __future__ import annotations

from app.services.rag.retriever import RetrievalResult


def format_context(results: list[RetrievalResult]) -> str:
    """Render retrieved DBT chunks into a context block for the system prompt.

    Returns an empty string when results is empty so the pipeline can skip
    the stage entirely rather than injecting a blank section.
    """
    if not results:
        return ""

    lines: list[str] = [
        "## Relevant DBT Knowledge",
        "",
        "The following excerpts from DBT skill handouts are relevant to the user's "
        "message. Draw on them naturally to inform your response. Do not quote them "
        "verbatim, reference handout numbers, or mention that you retrieved anything — "
        "just let the knowledge shape what you say.",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.handout_id} — {r.skill_category}")
        lines.append(r.text.strip())
        lines.append("")

    return "\n".join(lines).rstrip()

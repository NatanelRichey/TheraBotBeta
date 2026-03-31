"""TurnTrace — immutable record of one completed therapy pipeline turn.

Stored in two places:
  1. Redis  turn_trace:{session_id}:{turn_number}       (permanent, no TTL)
  2. JSONL  data/evals/traces.jsonl                     (append-only)

Stages in system_prompt are separated by "\\n\\n---\\n\\n" — split on that
to reconstruct individual pipeline stages for display.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TurnTrace(BaseModel):
    model_config = {"frozen": True}

    # ── Identifiers ───────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    turn_number: int
    profile: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    user_message: str

    # ── Pipeline ──────────────────────────────────────────────────────────────
    routing_path: list[str]
    cache_hit: bool
    # Derived by model_validator — strips label suffixes (:MISS, :gpt-4o, etc.)
    # and deduplicates, giving clean node names like ["cache_check", "gate_and_memory", ...]
    pipeline_stages: list[str]

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_label: Literal["SAFE", "REDIRECT", "CRISIS"] | None
    safety_confidence: float | None

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_required: bool
    rewritten_query: str | None
    retrieved_knowledge_count: int
    retrieved_skill_categories: list[str]
    # Full chunk data so the dashboard can render without a second lookup
    retrieved_chunks: list[dict]

    # ── Memory ────────────────────────────────────────────────────────────────
    had_working_memory: bool
    had_episodic: bool
    had_longterm: bool
    had_special_instructions: bool
    working_memory_text: str | None
    episodic_count: int
    longterm_count: int
    special_instructions: list[str]

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str | None
    llm_input_tokens: int | None
    llm_output_tokens: int | None
    llm_cost_usd: float | None

    # ── Quality ───────────────────────────────────────────────────────────────
    alpha_passed: bool | None
    alpha_flags: list[str]

    # ── Prompt ────────────────────────────────────────────────────────────────
    # Full assembled system prompt. Split on "\n\n---\n\n" to get individual stages.
    # None for cache-hit turns (llm_dispatch is never reached).
    system_prompt: str | None

    # ── Output ────────────────────────────────────────────────────────────────
    response: str

    @model_validator(mode="before")
    @classmethod
    def _derive_pipeline_stages(cls, data: dict) -> dict:
        """Derive clean stage names from the raw routing_path audit trail."""
        if not data.get("pipeline_stages"):
            path: list[str] = data.get("routing_path") or []
            seen: set[str] = set()
            stages: list[str] = []
            for entry in path:
                node = entry.split(":")[0]
                if node not in seen:
                    seen.add(node)
                    stages.append(node)
            data["pipeline_stages"] = stages
        return data

    @classmethod
    def from_state(cls, state: dict) -> "TurnTrace":
        """Build a TurnTrace from a (possibly augmented) AgentState dict.

        Callers pass {**state, "response": response} because state["response"]
        is not yet set at node execution time — LangGraph merges it after the
        node returns.
        """
        knowledge = state.get("retrieved_knowledge") or []
        episodic = state.get("episodic_chunks") or []
        longterm = state.get("longterm_gems") or []

        return cls(
            session_id=state["session_id"],
            user_id=state["user_id"],
            turn_number=state["turn_number"],
            profile=state.get("profile", "default"),
            user_message=state["user_message"],
            routing_path=list(state.get("routing_path") or []),
            cache_hit=state.get("cache_hit", False),
            pipeline_stages=[],  # filled by model_validator
            safety_label=state.get("safety_label"),
            safety_confidence=state.get("safety_confidence"),
            rag_required=state.get("rag_required", False),
            rewritten_query=state.get("rewritten_query"),
            retrieved_knowledge_count=len(knowledge),
            retrieved_skill_categories=list({c.skill_category for c in knowledge}),
            retrieved_chunks=[
                {
                    "text": c.text,
                    "handout_id": c.handout_id,
                    "module": c.module,
                    "skill_category": c.skill_category,
                    "score": c.score,
                }
                for c in knowledge
            ],
            had_working_memory=bool(state.get("working_memory")),
            had_episodic=bool(episodic),
            had_longterm=bool(longterm),
            had_special_instructions=bool(state.get("special_instructions")),
            working_memory_text=state.get("working_memory"),
            episodic_count=len(episodic),
            longterm_count=len(longterm),
            special_instructions=list(state.get("special_instructions") or []),
            llm_model=state.get("llm_model"),
            llm_input_tokens=state.get("llm_input_tokens"),
            llm_output_tokens=state.get("llm_output_tokens"),
            llm_cost_usd=state.get("llm_cost_usd"),
            alpha_passed=state.get("alpha_passed"),
            alpha_flags=list(state.get("alpha_flags") or []),
            system_prompt=state.get("system_prompt"),
            response=state.get("response") or "",
        )


class TurnTraceSummary(BaseModel):
    """Slim version used for sidebar/list endpoints — no prompt, no full response, no chunks."""

    session_id: str
    user_id: str
    turn_number: int
    profile: str
    timestamp: str
    cache_hit: bool
    pipeline_stages: list[str]
    safety_label: Literal["SAFE", "REDIRECT", "CRISIS"] | None
    rag_required: bool
    retrieved_knowledge_count: int
    llm_model: str | None
    llm_input_tokens: int | None
    llm_output_tokens: int | None
    llm_cost_usd: float | None
    alpha_passed: bool | None
    alpha_flags: list[str]
    user_message_preview: str  # first 120 chars
    response_preview: str      # first 120 chars

    @classmethod
    def from_trace(cls, trace: TurnTrace) -> "TurnTraceSummary":
        return cls(
            session_id=trace.session_id,
            user_id=trace.user_id,
            turn_number=trace.turn_number,
            profile=trace.profile,
            timestamp=trace.timestamp,
            cache_hit=trace.cache_hit,
            pipeline_stages=trace.pipeline_stages,
            safety_label=trace.safety_label,
            rag_required=trace.rag_required,
            retrieved_knowledge_count=trace.retrieved_knowledge_count,
            llm_model=trace.llm_model,
            llm_input_tokens=trace.llm_input_tokens,
            llm_output_tokens=trace.llm_output_tokens,
            llm_cost_usd=trace.llm_cost_usd,
            alpha_passed=trace.alpha_passed,
            alpha_flags=trace.alpha_flags,
            user_message_preview=trace.user_message[:120],
            response_preview=trace.response[:120],
        )

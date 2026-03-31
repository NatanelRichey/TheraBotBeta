"""LangGraph agent state for the therapy orchestration pipeline.

Every node reads from and writes a partial update to AgentState.
Fields with ``Annotated[list, operator.add]`` are append-only reducers —
multiple nodes can safely extend them without overwriting each other.
All other fields are last-write-wins.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from app.models.chat import ChatMessage


# ── Supporting data classes ───────────────────────────────────────────────────


@dataclass
class EpisodicChunk:
    """A past conversation turn retrieved from the per-user episodic vector store."""

    text: str
    turn_number: int
    timestamp: str
    score: float
    topic_tags: list[str] = field(default_factory=list)
    rag_triggered: bool = False
    skill_categories: list[str] = field(default_factory=list)


@dataclass
class KnowledgeChunk:
    """A DBT knowledge chunk retrieved from the shared knowledge vector store."""

    text: str
    handout_id: str
    module: str
    skill_category: str
    score: float


@dataclass
class LongtermGem:
    """A curated long-term memory entry from the per-user lt_{user_id} collection.

    Types:
        personal_fact        — factual life information (family, location, job, etc.)
        sensitive_disclosure — carefully summarized trauma or intimate disclosure
        psych_pattern        — recurring emotional/behavioural pattern (stored, not injected)
    """

    text: str
    memory_type: str  # personal_fact | sensitive_disclosure | psych_pattern
    score: float
    turn_number: int | None = None
    timestamp: str = ""


# ── Agent state ───────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    # ── Request ──────────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    user_message: str
    turn_number: int
    session_messages: list[ChatMessage]  # windowed history (last N messages)
    profile: str  # "default" | "cheap"

    # ── GateAgent outputs ────────────────────────────────────────────────────
    safety_label: Literal["SAFE", "REDIRECT", "CRISIS"] | None
    safety_confidence: float | None
    rag_required: bool

    # ── Sensitive topic flag ─────────────────────────────────────────────────
    sensitive_flag_active: bool          # flag is currently raised
    sensitive_flag_dropping: bool        # topic just shifted — flag about to drop
    sensitive_flag_raised_at_turn: int | None  # turn when flag was raised

    # ── Personal info detection ──────────────────────────────────────────────
    personal_info_detected: bool

    # ── MemoryAgent outputs ──────────────────────────────────────────────────
    working_memory: str | None           # two-tier summary (backlog + current block)
    episodic_chunks: list[EpisodicChunk] # top-k similar past turns from episodic store
    longterm_gems: list[LongtermGem]     # retrieved personal_fact + sensitive_disclosure
    rag_trigger_count: int               # tracks toward psych profile threshold

    # ── Special instructions (Stage 5 of prompt pipeline) ───────────────────
    special_instructions: list[str]      # high-priority override instructions

    # ── RAG outputs (populated only when rag_required=True) ─────────────────
    retrieved_knowledge: list[KnowledgeChunk]
    rewritten_query: str | None

    # ── LLM dispatch ─────────────────────────────────────────────────────────
    raw_response: str | None
    llm_model: str | None
    llm_input_tokens: int | None
    llm_output_tokens: int | None
    llm_cost_usd: float | None
    system_prompt: str | None   # full assembled system prompt (all stages joined)

    # ── AlphaAgent outputs ───────────────────────────────────────────────────
    alpha_passed: bool | None
    alpha_flags: Annotated[list[str], operator.add]  # append-only

    # ── Final ────────────────────────────────────────────────────────────────
    response: str | None
    routing_path: Annotated[list[str], operator.add]  # audit trail — append-only
    cache_hit: bool

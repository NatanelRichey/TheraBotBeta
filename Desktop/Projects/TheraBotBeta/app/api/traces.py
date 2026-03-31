"""Eval / debug API — exposes turn traces and memory layer data.

All endpoints are read-only. Intended for the /eval dashboard and CLI tooling.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_session_store
from app.core.logging import get_logger
from app.models.turn_trace import TurnTrace, TurnTraceSummary
from app.services.agents.memory import get_memory_agent
from app.services.cache.redis_client import get_redis_client
from app.services.turn_trace_store import get_turn_trace_store

router = APIRouter(prefix="/api/eval", tags=["eval"])
logger = get_logger(__name__)


# ── Traces ────────────────────────────────────────────────────────────────────


@router.get("/traces/{session_id}", response_model=list[TurnTraceSummary])
async def list_traces(session_id: str, last: int = 50) -> list[TurnTraceSummary]:
    """Return summary of all traces for a session, oldest-first."""
    store = get_turn_trace_store()
    traces = await store.list_for_session(session_id, last_n=last)
    return [TurnTraceSummary.from_trace(t) for t in traces]


@router.get("/traces/{session_id}/{turn_number}", response_model=TurnTrace)
async def get_trace(session_id: str, turn_number: int) -> TurnTrace:
    """Return full trace for a specific turn."""
    store = get_turn_trace_store()
    trace = await store.get(session_id, turn_number)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.get("/users/{user_id}/traces", response_model=list[TurnTraceSummary])
async def list_user_traces(user_id: str, last: int = 100) -> list[TurnTraceSummary]:
    """Return recent trace summaries across all sessions for a user."""
    store = get_turn_trace_store()
    traces = await store.list_for_user(user_id, last_n=last)
    return [TurnTraceSummary.from_trace(t) for t in traces]


# ── Sessions ──────────────────────────────────────────────────────────────────


@router.get("/users/{user_id}/sessions")
async def list_sessions(user_id: str) -> list[dict[str, Any]]:
    """Return all sessions for a user from the Redis session store."""
    try:
        uid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid user_id UUID")
    store = get_session_store()
    summaries = await store.list_for_user(uid)
    return [
        {
            "id": str(s.id),
            "user_id": str(s.user_id),
            "message_count": s.message_count,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in summaries
    ]


# ── Memory ────────────────────────────────────────────────────────────────────


@router.get("/memory/{user_id}/{session_id}")
async def get_memory(user_id: str, session_id: str) -> dict[str, Any]:
    """Return all memory layer data for a user+session.

    Sources:
      - session_messages       Redis sess:{session_id}:msgs
      - working_memory_current Redis wm_current:{session_id}
      - working_memory_backlog Redis wm_backlog:{session_id}
      - sensitive_flag         Redis sensitive_flag:{session_id}
      - semantic_cache         Redis semantic_cache:{session_id}
      - episodic_chunks        ChromaDB ep_{user_id} — all stored turns
      - longterm_gems          ChromaDB lt_{user_id} — all stored gems
    """
    try:
        uid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id UUID")

    redis = get_redis_client()
    memory_agent = get_memory_agent()

    # ── Session messages from Redis session store ──────────────────────────
    session_store = get_session_store()
    session = await session_store.get(uid)
    session_messages: list[dict] = []
    if session:
        session_messages = [
            {
                "id": str(m.id),
                "role": m.role.value,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "model": m.model,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "cost_usd": m.cost_usd,
            }
            for m in session.messages
        ]

    # ── Working memory from Redis ──────────────────────────────────────────
    wm_current_raw = await redis._client.get(f"wm_current:{session_id}")
    wm_backlog_raw = await redis._client.get(f"wm_backlog:{session_id}")
    working_memory_current: str | None = None
    working_memory_backlog: str | None = None
    if wm_current_raw:
        try:
            wm_current_working = json.loads(wm_current_raw)
            working_memory_current = wm_current_working.get("summary")
        except (json.JSONDecodeError, AttributeError):
            working_memory_current = wm_current_raw
    if wm_backlog_raw:
        try:
            wm_backlog_data = json.loads(wm_backlog_raw)
            working_memory_backlog = wm_backlog_data.get("summary")
        except (json.JSONDecodeError, AttributeError):
            working_memory_backlog = wm_backlog_raw

    # ── Sensitive flag from Redis ──────────────────────────────────────────
    sensitive_raw = await redis._client.get(f"sensitive_flag:{session_id}")
    sensitive_flag: dict | None = None
    if sensitive_raw:
        try:
            sensitive_flag = json.loads(sensitive_raw)
        except json.JSONDecodeError:
            sensitive_flag = {"raw": sensitive_raw}

    # ── Semantic cache entries from Redis ──────────────────────────────────
    cache_raw = await redis._client.lrange(f"semantic_cache:{session_id}", 0, -1)
    semantic_cache_entries: list[dict] = []
    for entry in cache_raw:
        try:
            parsed = json.loads(entry)
            # Don't expose raw embeddings — strip them for the dashboard
            semantic_cache_entries.append({
                "response": parsed.get("response", ""),
                "timestamp": parsed.get("timestamp", ""),
            })
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Episodic store — all turns for this user ───────────────────────────
    episodic_chunks: list[dict] = []
    try:
        col = memory_agent._episodic_collection(user_id.replace("-", ""))
        if col.count() > 0:
            results = col.get(include=["documents", "metadatas"])
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            for doc, meta in zip(docs, metas):
                episodic_chunks.append({"text": doc, **(meta or {})})
        # Sort by turn_number ascending
        episodic_chunks.sort(key=lambda x: int(x.get("turn_number", 0)))
    except Exception as exc:
        logger.warning("episodic_fetch_failed", user_id=user_id, error=str(exc))

    # ── Long-term gems — all gems for this user ────────────────────────────
    longterm_gems: list[dict] = []
    try:
        lt_col = memory_agent._longterm_collection(user_id.replace("-", ""))
        if lt_col.count() > 0:
            lt_results = lt_col.get(include=["documents", "metadatas"])
            lt_docs = lt_results.get("documents") or []
            lt_metas = lt_results.get("metadatas") or []
            for doc, meta in zip(lt_docs, lt_metas):
                longterm_gems.append({"text": doc, **(meta or {})})
        # Sort by memory_type then turn_number
        longterm_gems.sort(key=lambda x: (x.get("memory_type", ""), int(x.get("turn_number", 0))))
    except Exception as exc:
        logger.warning("longterm_fetch_failed", user_id=user_id, error=str(exc))

    return {
        "session_messages": session_messages,
        "working_memory_current": working_memory_current,
        "working_memory_backlog": working_memory_backlog,
        "sensitive_flag": sensitive_flag,
        "semantic_cache_entries": semantic_cache_entries,
        "episodic_chunks": episodic_chunks,
        "longterm_gems": longterm_gems,
    }

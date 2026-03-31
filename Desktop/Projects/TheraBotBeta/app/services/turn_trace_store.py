"""TurnTraceStore — persists per-turn pipeline traces.

Write targets:
  1. Redis  turn_trace:{session_id}:{turn_number}        no TTL (permanent)
  2. Redis  turn_traces_sess:{session_id}                sorted set, score=epoch (permanent)
  3. Redis  turn_traces:{user_id}                        sorted set, score=epoch (permanent)
  4. JSONL  data/evals/traces.jsonl                      append-only

All writes are fire-and-forget via asyncio.create_task() — they must not
raise exceptions that would surface to the caller.

Note: _write_redis accesses self._redis._client directly to use a pipeline
for atomicity. This intentionally bypasses the RedisClient wrapper's JSON
serialization layer since the trace is already a pre-serialized JSON string.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.turn_trace import TurnTrace
from app.services.cache.redis_client import RedisClient, get_redis_client

logger = get_logger(__name__)

# Module-level lock prevents concurrent JSONL writes from interleaving
_jsonl_lock = asyncio.Lock()


class TurnTraceStore:
    def __init__(self, redis: RedisClient, jsonl_path: Path) -> None:
        self._redis = redis
        self._jsonl_path = jsonl_path
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _trace_key(self, session_id: str, turn_number: int) -> str:
        return f"turn_trace:{session_id}:{turn_number}"

    def _sess_index_key(self, session_id: str) -> str:
        return f"turn_traces_sess:{session_id}"

    def _user_index_key(self, user_id: str) -> str:
        return f"turn_traces:{user_id}"

    # ── Public API ────────────────────────────────────────────────────────────

    async def write(self, trace: TurnTrace) -> None:
        """Write trace to Redis + JSONL. Safe to call via asyncio.create_task()."""
        try:
            await asyncio.gather(
                self._write_redis(trace),
                self._write_jsonl(trace),
            )
            logger.debug(
                "turn_trace_written",
                session_id=trace.session_id,
                turn=trace.turn_number,
            )
        except Exception as exc:
            logger.warning(
                "turn_trace_write_failed",
                session_id=trace.session_id,
                turn_number=trace.turn_number,
                error=str(exc),
            )

    async def get(self, session_id: str, turn_number: int) -> TurnTrace | None:
        """Retrieve a single trace. Returns None if not found."""
        key = self._trace_key(session_id, turn_number)
        raw = await self._redis._client.get(key)
        if raw is None:
            return None
        return TurnTrace.model_validate_json(raw)

    async def list_for_session(
        self, session_id: str, last_n: int = 50
    ) -> list[TurnTrace]:
        """Retrieve the last N traces for a session, ordered oldest-first."""
        index_key = self._sess_index_key(session_id)
        # zrevrange returns highest scores (most recent) first; we reverse at end
        members = await self._redis._client.zrevrange(index_key, 0, last_n - 1)
        traces = await self._batch_get(members)
        return list(reversed(traces))  # oldest-first for display

    async def list_for_user(
        self, user_id: str, last_n: int = 100
    ) -> list[TurnTrace]:
        """Retrieve recent traces across all sessions for a user, oldest-first."""
        index_key = self._user_index_key(user_id)
        members = await self._redis._client.zrevrange(index_key, 0, last_n - 1)
        traces = await self._batch_get(members)
        return list(reversed(traces))

    # ── Internal writes ───────────────────────────────────────────────────────

    async def _write_redis(self, trace: TurnTrace) -> None:
        trace_key = self._trace_key(trace.session_id, trace.turn_number)
        sess_key = self._sess_index_key(trace.session_id)
        user_key = self._user_index_key(trace.user_id)
        member = f"{trace.session_id}:{trace.turn_number}"
        score = datetime.fromisoformat(trace.timestamp).timestamp()
        payload = trace.model_dump_json()

        pipe = self._redis._client.pipeline()
        pipe.set(trace_key, payload)          # permanent — no TTL
        pipe.zadd(sess_key, {member: score})  # permanent — no TTL
        pipe.zadd(user_key, {member: score})  # permanent — no TTL
        await pipe.execute()

    async def _write_jsonl(self, trace: TurnTrace) -> None:
        line = trace.model_dump_json() + "\n"
        async with _jsonl_lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    async def _batch_get(self, members: list[str]) -> list[TurnTrace]:
        """Fetch multiple traces by sorted-set member strings ({session_id}:{turn})."""
        traces: list[TurnTrace] = []
        for member in members:
            # member format: "{session_id}:{turn_number}"
            last_colon = member.rfind(":")
            sid = member[:last_colon]
            turn = int(member[last_colon + 1:])
            t = await self.get(sid, turn)
            if t is not None:
                traces.append(t)
        return traces


@lru_cache
def get_turn_trace_store() -> TurnTraceStore:
    """Singleton factory. Same pattern as get_therapy_agent()."""
    settings = get_settings()
    return TurnTraceStore(
        redis=get_redis_client(),
        jsonl_path=Path(settings.turn_trace_jsonl_path),
    )

"""TherapyAgent — LangGraph orchestrator for the full therapy pipeline.

Graph topology
──────────────
START
  └─► cache_check ──── HIT ──► cache_reformulate (cheap LLM) ──────────► END
              │
             MISS
              ▼
        gate_and_memory          ← 7-way concurrent fan-out
              │                    GateAgent (safety, rag, sensitive, personal_info)
              │                    + sensitive flag load/topic-shift check
              │                    + MemoryAgent (working_memory, episodic, longterm gems)
       route_safety
              │
    ┌─────────┼──────────────────┐
  CRISIS      │               SAFE / REDIRECT
    │    SAFE/REDIRECT+rag       │  no rag
    ▼         ▼                  ▼
crisis_node  rag_retrieve    llm_dispatch
               │                 │
               └────────────────►│
                            llm_dispatch  ← system prompt: stages 1-5
                                          (identity, knowledge, memory, safety,
                                           format, special instructions)
                                 │
                           alpha_check
                          pass │   │ fail
                               ▼   ▼
                          cache_write  fallback_node
                          (expanded: also writes episodic, fires background
                           memory tasks — personal fact, sensitive disclosure,
                           psych pattern extraction, working memory rebuild)
                               │           │
                               └─────┬─────┘
                                     ▼
                                    END

routing_path is the audit trail — every node appends its name.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.models.agent_state import AgentState, KnowledgeChunk
from app.models.turn_trace import TurnTrace
from app.models.chat import ChatMessage, MessageRole
from app.services.agents.alpha_agent import AlphaAgent
from app.services.agents.gate_agent import GateAgent
from app.services.agents.memory import MemoryAgent
from app.services.cache.semantic_cache import SemanticCache
from app.services.turn_trace_store import TurnTraceStore
from app.services.prompts.pipeline import assemble
from app.services.rag.grounding import format_context, format_episodic_context, format_longterm_context
from app.services.rag.retriever import KnowledgeRetriever, RetrievalResult
from app.services.llm.router import LLMRouter

logger = get_logger(__name__)

_CRISIS_RESPONSE = (
    "I'm really concerned about what you've shared. Please reach out to a crisis "
    "helpline right now — you can call or text 988 (Suicide & Crisis Lifeline) "
    "anytime, 24/7. You don't have to go through this alone."
)

_CACHE_REFORMULATE_PROMPT = """\
You are a warm wellness companion. The user has returned to something you've \
explored together before.

Your previous response was:
{cached_response}

Briefly acknowledge you've touched on this together, gently reference the \
essence of your prior answer, and warmly ask whether something has changed \
or if they'd like to go deeper somewhere specific.

Keep it natural — 2 to 3 sentences. Do not sound robotic or scripted."""

_PERSONAL_INFO_INSTRUCTION = (
    "The user just shared personal information about their life. "
    "Gently explore this further with one warm, natural follow-up question "
    "before moving on — show genuine curiosity, not interrogation."
)

_SENSITIVE_LISTENING_INSTRUCTION = (
    "SENSITIVE TOPIC ACTIVE: The user is sharing something deeply personal. "
    "Listen fully and hold space. Do not probe, redirect, offer solutions, "
    "or ask follow-up questions. Reflect warmly and stay present with them."
)


class TherapyAgent:
    """LangGraph orchestrator — wires all agents, memory, cache, and RAG."""

    def __init__(
        self,
        gate: GateAgent,
        memory: MemoryAgent,
        alpha: AlphaAgent,
        cache: SemanticCache,
        retriever: KnowledgeRetriever,
        dispatch_router: LLMRouter,
        cheap_router: LLMRouter,
        trace_store: TurnTraceStore,
    ) -> None:
        self._gate = gate
        self._memory = memory
        self._alpha = alpha
        self._cache = cache
        self._retriever = retriever
        self._router = dispatch_router
        self._cheap_router = cheap_router
        self._trace_store = trace_store
        self._graph = self._build_graph()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def _cache_check_node(self, state: AgentState) -> dict[str, Any]:
        cached = await self._cache.get(state["session_id"], state["user_message"])
        if cached:
            # Rather than returning verbatim, acknowledge the repetition warmly.
            reformulated = await self._reformulate_cached(
                cached, state["user_message"]
            )
            asyncio.create_task(
                self._trace_store.write(
                    TurnTrace.from_state({
                        **state,
                        "response": reformulated,
                        "cache_hit": True,
                        "routing_path": ["cache_check:HIT"],
                    })
                )
            )
            return {
                "response": reformulated,
                "cache_hit": True,
                "routing_path": ["cache_check:HIT"],
            }
        return {
            "cache_hit": False,
            "routing_path": ["cache_check:MISS"],
        }

    async def _reformulate_cached(
        self, cached_response: str, user_message: str
    ) -> str:
        """Use cheap LLM to acknowledge a repeated question naturally."""
        system = _CACHE_REFORMULATE_PROMPT.format(cached_response=cached_response)
        try:
            resp = await self._cheap_router.complete(
                [
                    ChatMessage(role=MessageRole.system, content=system),
                    ChatMessage(role=MessageRole.user, content=user_message),
                ],
                max_tokens=120,
                temperature=0.7,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning("cache_reformulate_failed", error=str(exc))
            return cached_response  # fall back to original if reformulation fails

    async def _gate_and_memory_node(self, state: AgentState) -> dict[str, Any]:
        """7-way concurrent fan-out: classifiers + memory loads."""
        msg = state["user_message"]
        session_id = state["session_id"]
        user_id = state["user_id"]

        # Load sensitive flag state from Redis (needed before gathering)
        flag_data = await self._memory.load_sensitive_flag(session_id)
        flag_was_active = flag_data is not None
        raised_at_turn = flag_data["raised_at_turn"] if flag_data else None

        # 7 concurrent tasks
        (
            (safety_label, safety_confidence),
            rag_required,
            is_sensitive,
            personal_info_detected,
            working_memory,
            episodic_chunks,
            longterm_gems,
        ) = await asyncio.gather(
            self._gate.classify_safety(msg),
            self._gate.classify_rag(msg),
            self._gate.classify_sensitive(msg),
            self._gate.classify_personal_info(msg),
            self._memory.load_working_memory(
                session_id=session_id,
                current_turn=state["turn_number"],
                recent_messages=state.get("session_messages", []),
            ),
            self._memory.load_episodic(
                user_id=user_id,
                current_message=msg,
            ),
            self._memory.load_longterm_gems(
                user_id=user_id,
                current_message=msg,
            ),
        )

        # ── Sensitive flag logic ──────────────────────────────────────────────
        sensitive_flag_active = False
        sensitive_flag_dropping = False

        if is_sensitive and not flag_was_active:
            # New sensitive topic — raise flag
            asyncio.create_task(
                self._memory.set_sensitive_flag(session_id, state["turn_number"])
            )
            sensitive_flag_active = True
            raised_at_turn = state["turn_number"]
        elif is_sensitive and flag_was_active:
            # Still in the sensitive window
            sensitive_flag_active = True
        elif not is_sensitive and flag_was_active:
            # Potentially shifting away — check with topic-shift classifier
            shifted = await self._gate.classify_topic_shift(msg)
            if shifted:
                sensitive_flag_dropping = True
                asyncio.create_task(self._memory.clear_sensitive_flag(session_id))
            else:
                sensitive_flag_active = True  # still in window

        # ── Build special instructions ────────────────────────────────────────
        special_instructions: list[str] = []
        if personal_info_detected:
            special_instructions.append(_PERSONAL_INFO_INSTRUCTION)
        if sensitive_flag_active and not sensitive_flag_dropping:
            special_instructions.append(_SENSITIVE_LISTENING_INSTRUCTION)

        routing_tag = (
            f"gate_and_memory:safety={safety_label},"
            f"rag={rag_required},"
            f"sensitive={'dropping' if sensitive_flag_dropping else sensitive_flag_active},"
            f"personal_info={personal_info_detected}"
        )

        return {
            "safety_label": safety_label,
            "safety_confidence": safety_confidence,
            "rag_required": rag_required,
            "sensitive_flag_active": sensitive_flag_active,
            "sensitive_flag_dropping": sensitive_flag_dropping,
            "sensitive_flag_raised_at_turn": raised_at_turn,
            "personal_info_detected": personal_info_detected,
            "working_memory": working_memory,
            "episodic_chunks": episodic_chunks,
            "longterm_gems": longterm_gems,
            "special_instructions": special_instructions,
            "routing_path": [routing_tag],
        }

    async def _rag_retrieve_node(self, state: AgentState) -> dict[str, Any]:
        try:
            results, rewritten_query = await self._retriever.retrieve_with_query(
                state["user_message"]
            )
            knowledge_chunks = [
                KnowledgeChunk(
                    text=r.text,
                    handout_id=r.handout_id,
                    module=r.module,
                    skill_category=r.skill_category,
                    score=r.score,
                )
                for r in results
            ]
            if knowledge_chunks:
                asyncio.create_task(
                    self._memory.record_rag_trigger(
                        user_id=state["user_id"],
                        rewritten_query=rewritten_query,
                        skill_categories=[c.skill_category for c in knowledge_chunks],
                    )
                )
            return {
                "retrieved_knowledge": knowledge_chunks,
                "rewritten_query": rewritten_query,
                "rag_trigger_count": state.get("rag_trigger_count", 0) + 1,
                "routing_path": [f"rag_retrieve:{len(knowledge_chunks)}_chunks"],
            }
        except Exception as exc:
            logger.warning("rag_retrieve_node_failed", error=str(exc))
            return {
                "retrieved_knowledge": [],
                "rewritten_query": None,
                "routing_path": ["rag_retrieve:FAILED"],
            }

    async def _llm_dispatch_node(self, state: AgentState) -> dict[str, Any]:
        retrieval_results = [
            RetrievalResult(
                text=c.text,
                handout_id=c.handout_id,
                module=c.module,
                skill_category=c.skill_category,
                content_type="",
                score=c.score,
            )
            for c in state.get("retrieved_knowledge", [])
        ]
        knowledge_context = format_context(retrieval_results)
        episodic_context = format_episodic_context(state.get("episodic_chunks", []))
        longterm_context = format_longterm_context(state.get("longterm_gems", []))

        system_prompt = assemble(
            knowledge_context=knowledge_context,
            working_memory=state.get("working_memory"),
            episodic_context=episodic_context,
            longterm_gems=longterm_context,
            special_instructions=state.get("special_instructions") or [],
            escalation=state.get("safety_label") == "REDIRECT",
        )

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.system, content=system_prompt),
            *state.get("session_messages", []),
        ]

        router = self._cheap_router if state.get("profile") == "cheap" else self._router
        try:
            response = await router.complete(messages)
            return {
                "raw_response": response.content,
                "llm_model": response.model,
                "llm_input_tokens": response.input_tokens,
                "llm_output_tokens": response.output_tokens,
                "llm_cost_usd": response.cost_usd,
                "system_prompt": system_prompt,
                "routing_path": [f"llm_dispatch:{response.model}"],
            }
        except Exception as exc:
            logger.error("llm_dispatch_node_failed", error=str(exc))
            return {
                "raw_response": self._alpha.fallback_response,
                "llm_model": None,
                "llm_input_tokens": None,
                "llm_output_tokens": None,
                "llm_cost_usd": None,
                "system_prompt": system_prompt,
                "routing_path": ["llm_dispatch:FAILED"],
            }

    async def _alpha_check_node(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("raw_response") or ""
        passed, flags = await self._alpha.check(
            user_message=state["user_message"],
            response=raw,
            session_messages=state.get("session_messages"),
            working_memory=state.get("working_memory"),
            longterm_gems=state.get("longterm_gems"),
            episodic_chunks=state.get("episodic_chunks"),
        )
        return {
            "alpha_passed": passed,
            "alpha_flags": flags,
            "routing_path": [f"alpha_check:{'PASS' if passed else 'FAIL'}"],
        }

    async def _cache_write_node(self, state: AgentState) -> dict[str, Any]:
        response = state.get("raw_response") or ""
        session_id = state["session_id"]
        user_id = state["user_id"]
        turn_number = state["turn_number"]

        # Semantic cache write — fire and forget
        asyncio.create_task(
            self._cache.set(session_id, state["user_message"], response)
        )

        # Episodic write — fire and forget
        skill_cats = [c.skill_category for c in state.get("retrieved_knowledge", [])]
        asyncio.create_task(
            self._memory.write_episodic(
                user_id=user_id,
                turn_number=turn_number,
                user_message=state["user_message"],
                assistant_response=response,
                rag_triggered=state.get("rag_required", False),
                skill_categories=skill_cats,
                sensitive_flag=state.get("sensitive_flag_active", False),
                personal_info_detected=state.get("personal_info_detected", False),
                routing_path=state.get("routing_path", []),
                llm_model=state.get("llm_model"),
                llm_input_tokens=state.get("llm_input_tokens"),
                llm_output_tokens=state.get("llm_output_tokens"),
                llm_cost_usd=state.get("llm_cost_usd"),
                alpha_flags=state.get("alpha_flags", []),
                safety_label=state.get("safety_label"),
            )
        )

        # Personal info extraction — fire and forget
        if state.get("personal_info_detected"):
            asyncio.create_task(
                self._memory.extract_personal_info(
                    user_id=user_id,
                    turn_number=turn_number,
                    user_message=state["user_message"],
                )
            )

        # Sensitive disclosure write — fire and forget when flag is dropping
        if state.get("sensitive_flag_dropping"):
            raised_at = state.get("sensitive_flag_raised_at_turn")
            if raised_at is not None:
                window = state.get("session_messages", [])
                # Slice from the turn when the flag was raised
                # session_messages is 0-indexed; turn_number is 1-indexed
                window_start = max(0, raised_at - 1)
                sensitive_window = window[window_start:]
                asyncio.create_task(
                    self._memory.write_sensitive_disclosure(
                        user_id=user_id,
                        session_id=session_id,
                        window_messages=sensitive_window,
                        raised_at_turn=raised_at,
                    )
                )

        # Working memory rebuild + psych pattern extraction — at every N turns
        if turn_number % self._memory._rebuild_interval == 0:
            session_msgs = state.get("session_messages", [])
            asyncio.create_task(
                self._memory.extract_psych_pattern(
                    user_id=user_id,
                    session_id=session_id,
                    messages=session_msgs,
                )
            )

        asyncio.create_task(
            self._trace_store.write(TurnTrace.from_state({**state, "response": response}))
        )

        return {
            "response": response,
            "routing_path": ["cache_write"],
        }

    async def _crisis_node(self, state: AgentState) -> dict[str, Any]:
        # routing_path in state is the full audit trail up to this node;
        # "crisis_response" label is in the return dict and not yet merged — acceptable.
        asyncio.create_task(
            self._trace_store.write(
                TurnTrace.from_state({**state, "response": _CRISIS_RESPONSE})
            )
        )
        return {
            "response": _CRISIS_RESPONSE,
            "routing_path": ["crisis_response"],
        }

    async def _fallback_node(self, state: AgentState) -> dict[str, Any]:
        fallback = self._alpha.fallback_response
        asyncio.create_task(
            self._trace_store.write(
                TurnTrace.from_state({**state, "response": fallback})
            )
        )
        return {
            "response": fallback,
            "routing_path": ["fallback_response"],
        }

    # ── Routing conditions ────────────────────────────────────────────────────

    def _route_after_cache(self, state: AgentState) -> str:
        return END if state.get("cache_hit") else "gate_and_memory"

    def _route_after_gate(self, state: AgentState) -> str:
        if state.get("safety_label") == "CRISIS":
            return "crisis_node"
        return "rag_retrieve" if state.get("rag_required") else "llm_dispatch"

    def _route_after_alpha(self, state: AgentState) -> str:
        return "cache_write" if state.get("alpha_passed", True) else "fallback_node"

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)

        g.add_node("cache_check", self._cache_check_node)
        g.add_node("gate_and_memory", self._gate_and_memory_node)
        g.add_node("rag_retrieve", self._rag_retrieve_node)
        g.add_node("llm_dispatch", self._llm_dispatch_node)
        g.add_node("alpha_check", self._alpha_check_node)
        g.add_node("cache_write", self._cache_write_node)
        g.add_node("crisis_node", self._crisis_node)
        g.add_node("fallback_node", self._fallback_node)

        g.add_edge(START, "cache_check")
        g.add_conditional_edges(
            "cache_check",
            self._route_after_cache,
            {END: END, "gate_and_memory": "gate_and_memory"},
        )
        g.add_conditional_edges(
            "gate_and_memory",
            self._route_after_gate,
            {
                "crisis_node": "crisis_node",
                "rag_retrieve": "rag_retrieve",
                "llm_dispatch": "llm_dispatch",
            },
        )
        g.add_edge("rag_retrieve", "llm_dispatch")
        g.add_edge("llm_dispatch", "alpha_check")
        g.add_conditional_edges(
            "alpha_check",
            self._route_after_alpha,
            {"cache_write": "cache_write", "fallback_node": "fallback_node"},
        )
        g.add_edge("cache_write", END)
        g.add_edge("fallback_node", END)
        g.add_edge("crisis_node", END)

        return g.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    async def invoke(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        turn_number: int,
        session_messages: list[ChatMessage],
        profile: str = "default",
    ) -> tuple[str, list[str]]:
        """Run the therapy pipeline for one user turn.

        Returns:
            (response, routing_path) — final text and the node audit trail.
        """
        initial_state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_message,
            "turn_number": turn_number,
            "session_messages": session_messages,
            "profile": profile,
            "safety_label": None,
            "safety_confidence": None,
            "rag_required": False,
            "sensitive_flag_active": False,
            "sensitive_flag_dropping": False,
            "sensitive_flag_raised_at_turn": None,
            "personal_info_detected": False,
            "working_memory": None,
            "episodic_chunks": [],
            "longterm_gems": [],
            "special_instructions": [],
            "rag_trigger_count": 0,
            "retrieved_knowledge": [],
            "rewritten_query": None,
            "raw_response": None,
            "llm_model": None,
            "llm_input_tokens": None,
            "llm_output_tokens": None,
            "llm_cost_usd": None,
            "alpha_passed": None,
            "alpha_flags": [],
            "response": None,
            "routing_path": [],
            "cache_hit": False,
        }

        final_state = await self._graph.ainvoke(initial_state)
        response = final_state.get("response") or self._alpha.fallback_response
        routing_path: list[str] = final_state.get("routing_path", [])

        logger.info(
            "therapy_agent_complete",
            session_id=session_id,
            user_id=user_id,
            turn=turn_number,
            routing_path=routing_path,
            cache_hit=final_state.get("cache_hit", False),
            rag_triggered=final_state.get("rag_required", False),
            sensitive_active=final_state.get("sensitive_flag_active", False),
            personal_info=final_state.get("personal_info_detected", False),
        )

        return response, routing_path


@lru_cache
def get_therapy_agent() -> TherapyAgent:
    """Singleton factory — all sub-agents are themselves singletons via lru_cache."""
    from app.services.agents.alpha_agent import AlphaAgent
    from app.services.agents.gate_agent import GateAgent
    from app.services.agents.memory import get_memory_agent
    from app.services.cache.semantic_cache import get_semantic_cache
    from app.services.llm.router import get_cheap_router, get_router
    from app.services.rag.retriever import get_retriever
    from app.services.turn_trace_store import get_turn_trace_store

    cheap = get_cheap_router()
    return TherapyAgent(
        gate=GateAgent(router=cheap),
        memory=get_memory_agent(),
        alpha=AlphaAgent(router=cheap),
        cache=get_semantic_cache(),
        retriever=get_retriever(),
        dispatch_router=get_router(),
        cheap_router=cheap,
        trace_store=get_turn_trace_store(),
    )

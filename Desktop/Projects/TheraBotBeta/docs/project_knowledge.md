# Project Knowledge Base: AI Wellness Companion
## Role Target + Project Roadmap

---

## TARGET ROLE: Generative AI Engineer (Lead)

### Role Summary
Lead AI engineer at a B2C therapy startup. End-to-end ownership: infrastructure, production systems, internal adoption, cross-team enablement. Building a consumer therapy AI product.

### Key Tension To Demonstrate
Advocate for engineering quality (tech debt, best practices) WHILE maintaining a product mindset (customer-first, ship fast, iterate).

### Required Skills Checklist
- [x] ~1+ year experience with Generative AI / LLMs
- [x] Experience shipping GenAI features to production
- [x] Strong production-grade backend/software engineering (Python)
- [x] Experience with major AI providers (OpenAI, Anthropic, Google)
- [x] Prompt engineering and LLM behavior knowledge
- [x] Understanding Agents / multi-step workflows
- [x] Experience with AI frameworks (LangChain, LangGraph, etc.)
- [x] Familiarity with embeddings / vector databases
- [x] Problem-solving in fast-paced environments

### Bonus Skills Checklist
- [ ] MLOps / model deployment experience
- [x] Evaluation, monitoring, and guardrails (AlphaAgent output validation, TurnTrace observability dashboard)
- [ ] Fine-tuning or model customization
- [x] RAG systems (Retrieval-Augmented Generation)
- [x] Building retrieval pipelines and grounding LLM outputs

---

## PROJECT ARCHITECTURE OVERVIEW

```
app/
├── api/
│   ├── chat.py               # POST /chat, /chat/stream, /chat/compare, /chat/compare/vote
│   ├── evaluation.py
│   └── health.py
├── core/
│   ├── config.py             # pydantic-settings — all config via env vars
│   ├── logging.py            # structlog structured logging
│   └── dependencies.py       # DI container
├── models/
│   ├── agent_state.py        # LangGraph AgentState TypedDict + EpisodicChunk, KnowledgeChunk, LongtermGem
│   ├── chat.py               # ChatMessage, ChatRequest, ChatResponse, StreamChunk
│   └── session.py            # Session model
├── services/
│   ├── llm/
│   │   ├── base.py           # Abstract BaseLLMProvider
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── router.py         # Model routing + automatic fallback (default + cheap profiles)
│   │   └── cost_tracker.py   # Token usage + cost tracking
│   ├── prompts/
│   │   ├── templates.py      # Disk loader with in-memory cache
│   │   ├── pipeline.py       # 6-stage system prompt assembler
│   │   └── experiments.py    # ExperimentRunner — concurrent A/B variant execution
│   ├── rag/
│   │   ├── retriever.py      # LLM query rewrite + ChromaDB retrieval + score filtering
│   │   └── grounding.py      # format_context, format_episodic_context, format_longterm_context
│   ├── agents/
│   │   ├── therapy_agent.py  # LangGraph orchestrator (8-node graph)
│   │   ├── gate_agent.py     # 5-classifier gate (safety, RAG, sensitive, personal_info, topic_shift)
│   │   ├── alpha_agent.py    # Output validator (CLINICAL_CLAIM, HALLUCINATION, LOW_CONFIDENCE)
│   │   └── memory.py         # MemoryAgent — all 5 memory subsystems
│   ├── cache/
│   │   ├── semantic_cache.py # Session-scoped cosine similarity cache
│   │   └── redis_client.py   # Async Redis wrapper with JSON serialization
│   └── evaluation/
│       ├── evaluator.py
│       ├── metrics.py
│       └── dataset.py
├── session_store.py          # In-memory session store (swap point for Redis/DB)
└── main.py

data/
├── prompts/      # 6 .txt template files (identity ×3, format ×2, safety ×1)
├── chroma/       # DBT knowledge vector store (gitignored)
├── episodic/     # Per-user episodic turn history (gitignored)
├── longterm/     # Per-user long-term memory gems (gitignored)
└── knowledge/    # Raw DBT content (seeding source)

docs/             # Architecture reference documents
scripts/          # seed_knowledge.py, run_evals.py, fine_tune.py
tests/            # unit/, integration/, evals/
```

---

## MEMORY ARCHITECTURE (The Core Differentiator — Phase 5)

Six memory layers, shortest to longest lived:

| Layer | What | Storage | TTL | Scope |
|---|---|---|---|---|
| 1. Semantic Cache | Near-duplicate detection (≥0.92 cosine). On hit: LLM reformulates, acknowledges repetition. | Redis list `semantic_cache:{session_id}` | 1h | Session |
| 2. Raw History | Last 30 messages (sliding window) passed directly to LLM as chat turns. | In-memory SessionStore | Runtime | Session |
| 3. Working Memory | Two-tier: `wm_current` (last 20 turns compressed) + `wm_backlog` (merged older blocks). Rebuild every 20 turns. | Redis strings | 24h | Session |
| 4. Episodic Store | Every completed turn embedded in per-user ChromaDB. Top-3 similar past turns retrieved and injected each turn. | ChromaDB `ep_{user_id}` | Permanent | User |
| 5. Long-Term Gems | `personal_fact`, `sensitive_disclosure`, `psych_pattern` — extracted by LLM, stored in per-user ChromaDB. Facts + disclosures injected when semantically relevant. Psych patterns stored only. | ChromaDB `lt_{user_id}` | Permanent | User |
| 6. Special Instructions | Dynamic high-priority overrides injected as final stage (Stage 5) of system prompt. | AgentState (per-turn) | Per-turn | Turn |

### Sensitive Topic Flow
1. `classify_sensitive()` fires → Redis flag raised (`sensitive_flag:{session_id}`)
2. While active: Stage 5 gets listening-mode instruction ("hold space, do not probe")
3. `classify_topic_shift()` fires → flag drops → background LLM summarizes the sensitive window → writes `sensitive_disclosure` gem to ChromaDB
4. Future sessions: if similar topic comes up, the disclosure is retrieved and injected

### Personal Info Flow
1. `classify_personal_info()` fires → Stage 5 gets gentle follow-up instruction
2. After response: background LLM extracts fact as one sentence → writes `personal_fact` gem to ChromaDB
3. Future sessions: retrieved when semantically relevant → injected as "What I Know About You"

---

## LANGGRAPH GRAPH (8 nodes)

```
START → cache_check ── HIT → reformulate (cheap LLM) → END
                │
               MISS
                ▼
         gate_and_memory  (7-way asyncio.gather)
           safety | RAG gate | sensitive | personal_info |
           sensitive flag | working memory | episodic | longterm gems
                │
          route_safety
           CRISIS → crisis_node → END
           +rag → rag_retrieve → llm_dispatch
           no rag → llm_dispatch
                │
          llm_dispatch  (6-stage system prompt)
                │
          alpha_check → pass → cache_write → END
                      → fail → fallback_node → END
```

---

## SYSTEM PROMPT PIPELINE (6 Stages)

| Stage | Content | Condition |
|---|---|---|
| 1 | Identity template | Always |
| 2 | DBT knowledge (RAG chunks) | `rag_required=True` |
| 2.5 | Session memory + episodic + long-term gems | When data exists |
| 3 | Safety escalation | `safety_label=REDIRECT` |
| 4 | Format template | Always |
| **5** | **Special instructions** | **Any active override (highest priority)** |

---

## PHASE BREAKDOWN WITH SKILL MAPPING

### Phase 1: Foundation ✅ Complete
**Skills:** Python backend, AI provider integration, production infrastructure

- FastAPI + async throughout, session management
- LLM provider abstraction (OpenAI + Anthropic + DeepSeek + Kimi + OpenRouter)
- Provider routing with automatic fallback (`default` and `cheap` profiles)
- Token usage tracking + monthly budget hard stop
- Docker containerization

### Phase 2: Prompt Engineering ✅ Complete
**Skills:** Prompt engineering, LLM behavior, evaluation

- 6 versioned prompt templates (identity ×3, format ×2, safety ×1)
- Multi-stage system prompt pipeline
- A/B testing framework (ExperimentRunner — 4 types, asyncio.gather concurrency)
- POST /chat/compare + /chat/compare/vote endpoints

### Phase 3: RAG ✅ Complete
**Skills:** Embeddings, vector databases, RAG, retrieval pipelines

- ChromaDB knowledge store seeded with DBT handout content
- LLM query rewrite (user message → DBT terminology before embedding)
- Score-threshold filtering + grounding formatter
- RAG escalation gate — skips ChromaDB entirely for casual messages

### Phase 4: Agents & Workflows ✅ Complete
**Skills:** Agents, multi-step workflows, LangGraph, context management

- Full LangGraph graph with 8 nodes, safety routing, crisis short-circuit
- GateAgent (safety + RAG classifiers), AlphaAgent (output validation)
- MemoryAgent with episodic store + working memory
- Semantic cache wired as first node in graph

### Phase 5: Memory Architecture ✅ Complete
**Skills:** Stateful context management, multi-layered memory, production memory design

- Two-tier working memory (prevents summary bloat in long sessions)
- Long-term memory gems: personal facts, sensitive disclosures, psych patterns
- Sensitive topic detection with persistent Redis flag + careful summarization
- Personal info detection + gentle follow-up injection
- Stage 5 "Special Instructions" — highest-priority prompt overrides
- Session history sliding window (prevents unbounded context growth)
- Episodic context now injected into prompt (was a known gap)
- Semantic cache humanized (LLM reformulation instead of verbatim repeat)
- Psych profile removed from prompt injection (stored only, reserved for future)

### Phase 6: Evaluation, Monitoring & Guardrails ⏳ Planned
**Skills:** MLOps, evaluation pipelines, guardrails

- LLM-as-judge eval pipeline running in CI
- Input filter (off-topic, harmful content)
- Output filter (expanded from AlphaAgent)
- Quality degradation alerts, user feedback metrics
- Comprehensive audit logging

### Phase 7: Fine-Tuning ⏳ Planned
**Skills:** Fine-tuning, model customization

- Training data curation pipeline
- Fine-tuning execution (OpenAI API)
- Fine-tuned vs base model benchmark
- Cost-benefit analysis

### Phase 8: Persona Evolution (Future concept)
**Skills:** Adaptive AI, long-term personalization

- Psych patterns drive dynamic identity template selection
- Bot's feel and pacing evolve per user over time
- Nested/associative RAG (one memory triggers another)
- Admin-configurable special instructions from psych profile

---

## TALKING POINTS FOR INTERVIEWS

1. **"I built a multi-provider abstraction layer"** → Shows enterprise model integration + vendor diversity handling
2. **"I designed a 6-layer memory architecture"** → Shows production context management thinking; most devs just pass the full chat history
3. **"The semantic cache reformulates on hit instead of repeating verbatim"** → Shows product thinking, not just engineering
4. **"I built a sensitive topic detection system with a persistent flag"** → Shows responsible AI + stateful design across LangGraph invocations
5. **"Working memory uses a two-tier system to prevent summary bloat"** → Shows you thought about what happens at turn 200, not just turn 5
6. **"Long-term memory gems are stored in ChromaDB, retrieved by similarity when relevant"** → Shows RAG thinking applied to memory, not just knowledge bases
7. **"Psych patterns are stored but not injected — deliberately, to avoid over-profiling"** → Shows responsible AI reasoning and deliberate restraint
8. **"The LangGraph graph has an audit trail via routing_path — every node appends its name"** → Shows observability instinct
9. **"I implemented LLM-as-judge output validation before any response reaches the user"** → Shows guardrails awareness
10. **"The whole thing runs in Docker with structured logging and async throughout"** → Shows production engineering mindset

---

## KEY ARCHITECTURE GOTCHAS

- `session_messages` passed to TherapyAgent is already windowed (last 30) — the full history is in SessionStore
- The sensitive flag persists in Redis across LangGraph invocations (each invocation = one turn)
- Psych patterns are written to ChromaDB but **never** retrieved for prompt injection
- Episodic store has a score threshold of 0.60 — low-similarity chunks are filtered out
- Working memory rebuild fires at `turn_number % 20 == 0` — first rebuild at exactly turn 20
- `wm_current` summarizes only the last 20 messages; `wm_backlog` holds the merged older history
- ChromaDB metadata values must be str | int | float | bool — no lists or dicts
- The cheap router (DeepSeek/Kimi) handles all classifiers and memory operations; the dispatch router handles the actual therapy response

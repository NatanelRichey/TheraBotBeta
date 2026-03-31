# AI Wellness Companion — CLAUDE.md

## Project Overview
Production-grade AI conversational wellness companion. Demonstrates: multi-provider LLM integration, RAG, agents (LangGraph), prompt engineering, evaluation pipelines, guardrails, semantic caching, and a layered memory architecture.

This is a portfolio project for a Generative AI Engineer Lead role. Every design decision should reflect production thinking.

## Tech Stack
- **Backend:** FastAPI + Python 3.12 (async throughout)
- **LLM Providers:** OpenAI, Anthropic (behind abstraction layer)
- **Agent Framework:** LangGraph
- **Vector DB:** ChromaDB (dev), pgvector (prod)
- **Cache:** Redis (semantic caching + session memory)
- **Data Validation:** Pydantic v2
- **Testing:** pytest + pytest-asyncio
- **Containerization:** Docker + docker-compose
- **Linting:** ruff
- **Type Checking:** mypy

## Project Structure
```
app/                    → FastAPI application
  api/                  → Route handlers (thin — delegate to services)
    chat.py             → POST /chat, /chat/stream, /chat/compare, /chat/compare/vote
    traces.py           → GET /api/eval/* (traces, memory, sessions) — Phase 6a
  core/                 → Config, logging, dependency injection
  models/               → Pydantic schemas (request/response models)
    agent_state.py      → LangGraph AgentState TypedDict + supporting dataclasses
    turn_trace.py       → TurnTrace + TurnTraceSummary — per-turn pipeline record (Phase 6a)
  services/             → All business logic lives here
    llm/                → Provider abstraction, routing, cost tracking
    prompts/            → Template management, pipeline, A/B testing
      templates.py      → Disk loader with in-memory cache
      pipeline.py       → 6-stage system prompt assembler (identity, knowledge, memory, safety, format, special instructions)
      experiments.py    → ExperimentRunner — concurrent A/B variant execution
    rag/                → Embeddings, vector store, retrieval, grounding
      retriever.py      → Query rewrite + ChromaDB retrieval
      grounding.py      → Context formatters (DBT knowledge, episodic, long-term gems)
    agents/             → LangGraph agents, memory, safety routing
      therapy_agent.py  → Main LangGraph orchestrator (8-node graph)
      gate_agent.py     → 5-classifier gate (safety, RAG, sensitive, personal_info, topic_shift)
      alpha_agent.py    → Output validator (hallucination, clinical claims, low confidence)
      memory.py         → MemoryAgent — all memory subsystems
    cache/              → Semantic cache, Redis client
    turn_trace_store.py → TurnTraceStore — Redis + JSONL trace persistence (Phase 6a)
    evaluation/         → LLM-as-judge, metrics, eval datasets
static/                 → Single-file chat UI (index.html) — served at GET /
                          eval.html — debug dashboard — served at GET /eval (Phase 6a)
scripts/                → Utility scripts (seeding, evals, fine-tuning)
                          view_traces.py — CLI trace viewer (Phase 6a)
data/
  prompts/              → Prompt template .txt files (version-controlled)
  chroma/               → DBT knowledge vector store (gitignored)
  episodic/             → Per-user episodic turn history (gitignored)
  longterm/             → Per-user long-term memory gems (gitignored)
  evals/                → Eval datasets in JSONL format (gitignored)
docs/                   → Architecture reference documents
tests/                  → Unit, integration, and eval tests
```

## Conventions

### Code Style
- Type hints on ALL function signatures — no exceptions
- Pydantic v2 models for all data crossing boundaries (API, services, external)
- Async by default for anything I/O-bound
- Use `httpx.AsyncClient` for external API calls, never `requests`
- Structured logging with `structlog` — no print statements
- All config via environment variables through `pydantic-settings`

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_prefixed`

### Architecture Rules
- **Routes are thin.** API handlers validate input, call a service, return response. No business logic in routes.
- **Services are the brain.** All logic lives in `app/services/`.
- **One abstraction layer for LLMs.** Never call OpenAI or Anthropic directly from services — always go through `app/services/llm/`.
- **Prompts are data, not code.** Store prompt templates in `data/prompts/` as .txt files. Load them dynamically.
- **Every external call has a timeout and retry.** Use tenacity for retries with exponential backoff.
- **Errors are typed.** Custom exception classes in `app/core/exceptions.py`, caught by FastAPI exception handlers.

### Testing
- Run `pytest` before suggesting any PR or feature is complete
- Unit tests for all services (mock external APIs)
- Integration tests for API endpoints
- Eval tests for prompt quality (can be slow — separate test mark)
- Target: >80% coverage on services

### Security
- NEVER hardcode API keys. Always use environment variables.
- NEVER log user conversation content at INFO level (use DEBUG, and only in dev)
- All user input is validated through Pydantic models before processing
- `data/prompts/` is version-controlled (prompt templates are application code)
- `data/chroma/`, `data/episodic/`, `data/longterm/`, `data/evals/` are gitignored (runtime data)

### Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Each phase gets its own feature branch
- Squash merge to main

---

## Current Phase
**Active Phase: 6a complete — Turn Trace Store + Debug Dashboard**

---

## Memory Architecture (Phase 5)

Six memory layers, from shortest to longest lived:

### Layer 1 — Semantic Cache
- **What:** Within-session near-duplicate detection. Threshold: 0.92 cosine similarity.
- **New in Phase 5:** On hit, cheap LLM reformulates the cached response to acknowledge the repetition naturally instead of repeating verbatim.
- **Storage:** Redis list `semantic_cache:{session_id}` | TTL: 1h | Scope: session

### Layer 2 — Raw Session History (Sliding Window)
- **What:** The actual live conversation passed as chat messages to the LLM.
- **New in Phase 5:** Sliding window of last 30 messages. Older turns are covered by working memory.
- **Storage:** In-memory SessionStore dict | TTL: runtime | Scope: session

### Layer 3 — Two-Tier Working Memory
- **What:** LLM-compressed session narrative. Prevents context bloat across long sessions.
- **`wm_current:{session_id}`:** Summary of the most recent 20-turn block. Overwritten each rebuild.
- **`wm_backlog:{session_id}`:** Merged history of all prior 20-turn blocks. At each rebuild the old current is LLM-merged into the backlog before new current is written. Stays bounded because it merges compressed-into-compressed.
- **Rebuild interval:** Every 20 turns (config: `working_memory_rebuild_interval`).
- **Injection:** `## Session Memory` in stage 2.5 of the system prompt.
- **Storage:** Redis strings | TTL: 24h | Scope: session

### Layer 4 — Episodic Store
- **What:** Permanent per-user ChromaDB log of every completed turn. Cross-session recall.
- **Retrieval:** Top-3 most semantically similar past turns, score-filtered at ≥ 0.60.
- **New in Phase 5:** Wired into the prompt as `## Relevant Past Conversations` (was retrieved but never injected before). Richer metadata: `sensitive_flag`, `personal_info_detected`.
- **Storage:** ChromaDB `ep_{user_id}` at `data/episodic/` | TTL: permanent | Scope: per user

### Layer 5 — Long-Term Memory Gems (`lt_{user_id}`)
Three document types in a unified per-user ChromaDB collection at `data/longterm/`:

| Type | Trigger | Injected? |
|---|---|---|
| `personal_fact` | `classify_personal_info()` fires → background LLM extraction | Yes — `## What I Know About You` |
| `sensitive_disclosure` | Sensitive flag drops → careful LLM summary of sensitive window | Yes — `## What I Know About You` |
| `psych_pattern` | Every 20-turn rebuild (separate LLM call) + RAG trigger accumulation | **No — stored only** |

- **Storage:** ChromaDB `lt_{user_id}` at `data/longterm/` | TTL: permanent | Scope: per user

### Layer 6 — Special Instructions (Stage 5 Pipeline)
- **What:** Final system prompt section. Injected last so the model reads it with highest recency weight. Explicitly states it overrides all other guidelines.
- **Current uses:** personal info detected (gentle follow-up instruction), sensitive flag active (listening-mode instruction).
- **Future:** Admin-configurable overrides, persona-adaptation driven by psych profile.

---

## LangGraph Graph Topology

```
START
  └─► cache_check ── HIT ──► reformulate (cheap LLM) ──► END
              │
             MISS
              ▼
        gate_and_memory  (7-way asyncio.gather)
          ├── safety classification
          ├── RAG gate
          ├── sensitive topic classification        (Phase 5)
          ├── personal info detection               (Phase 5)
          ├── sensitive flag load/topic-shift check (Phase 5)
          ├── two-tier working memory load/rebuild  (Phase 5)
          ├── episodic retrieval
          └── long-term gem retrieval               (Phase 5)
              │
        route_safety
          ├── CRISIS → crisis_node → END
          ├── SAFE/REDIRECT + rag → rag_retrieve → llm_dispatch
          └── SAFE/REDIRECT no rag → llm_dispatch
              │
        llm_dispatch  (6-stage system prompt)
              Stage 1:  identity template
              Stage 2:  DBT knowledge context (if rag_required)
              Stage 2.5: session memory, episodic, long-term gems
              Stage 3:  safety escalation (if REDIRECT)
              Stage 4:  format template
              Stage 5:  special instructions (if any active)
              │
        alpha_check
          ├── pass → cache_write
          │    (fires: episodic write, personal fact extract, sensitive summary, psych pattern)
          └── fail → fallback_node → END
```

---

## RAG Architecture

Three vector stores:

```
knowledge_store  → static DBT/wellness content
                   shared across all users, seeded once
                   lives in: data/chroma/

episodic_store   → per-user conversation turns
                   one collection per user_id (ep_{user_id})
                   embedded and queried by semantic similarity
                   lives in: data/episodic/

longterm_store   → per-user curated memory gems
                   one collection per user_id (lt_{user_id})
                   personal_fact | sensitive_disclosure | psych_pattern
                   lives in: data/longterm/
```

Retrieval gate: `classify_rag()` runs before any ChromaDB query. Casual messages skip retrieval entirely.

---

## System Prompt Pipeline (6 Stages)

| Stage | Content | Condition |
|---|---|---|
| 1 | Identity template (warm/clinical) | Always |
| 2 | DBT knowledge context | `rag_required=True` |
| 2.5 | Session memory + episodic + long-term gems | When relevant data exists |
| 3 | Safety escalation block | `safety_label=REDIRECT` |
| 4 | Format template (short/long) | Always |
| 5 | Special instructions (highest priority) | When any instruction is active |

---

## Key Config Values (app/core/config.py)

| Key | Default | Purpose |
|---|---|---|
| `working_memory_rebuild_interval` | 20 | Turns between working memory rebuilds |
| `session_history_window` | 30 | Max raw messages passed to LLM per turn |
| `semantic_cache_threshold` | 0.92 | Cosine similarity for cache hit |
| `episodic_top_k` | 3 | Past turns retrieved per query |
| `psych_profile_rag_threshold` | 5 | RAG triggers before building psych pattern |
| `longterm_chroma_dir` | ./data/longterm | Long-term gems ChromaDB location |
| `episodic_chroma_dir` | ./data/episodic | Episodic ChromaDB location |
| `turn_trace_jsonl_path` | ./data/evals/traces.jsonl | Append-only JSONL trace log |

---

## Key Commands
```bash
# Run the app
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Seed vector database (upsert — safe to re-run)
python scripts/seed_knowledge.py

# Seed vector database (reset collection first)
python scripts/seed_knowledge.py --reset

# Run evaluation benchmark
python scripts/run_evals.py

# View turn traces (CLI)
python scripts/view_traces.py --session-id <uuid>
python scripts/view_traces.py --user-id <uuid> --last 30
python scripts/view_traces.py --session-id <uuid> --turn 3 --show-prompt

# Open debug dashboard
# http://localhost:8000/eval
```

---

## LLM Provider Notes
- OpenAI: GPT-4o (primary conversation), text-embedding-3-small (embeddings)
- Anthropic: Claude Sonnet (fallback conversation, safety checks)
- DeepSeek: OpenAI-compatible — reuses `OpenAIProvider` with `base_url="https://api.deepseek.com"`
- Kimi (Moonshot): OpenAI-compatible — reuses `OpenAIProvider` with `base_url="https://api.moonshot.cn/v1"`
- OpenRouter: OpenAI-compatible aggregator — one key gives access to all models. When `OPENROUTER_API_KEY` is set, both routing profiles use OpenRouter instead of direct provider keys.

### Routing profiles
| Profile | Direct keys | Via OpenRouter |
|---|---|---|
| `default` | `gpt-4o` → `claude-sonnet-4-6` | `openai/gpt-4o` → `anthropic/claude-sonnet-4-5` |
| `cheap` | `deepseek-chat` → `moonshot-v1-8k` | `deepseek/deepseek-chat` → `moonshot/moonshot-v1-8k` |

---

## Phase Notes

### Phase 2 — Prompt Engineering (complete)
- 6 .txt prompt templates, disk loader with in-memory cache
- 4-stage pipeline assembler
- ExperimentRunner with 4 experiment types, asyncio.gather concurrency
- POST /chat/compare and /chat/compare/vote endpoints

### Phase 3 — RAG (complete)
- ChromaDB knowledge store seeded with DBT handout content
- Query rewrite using DBT terminology via cheap LLM
- Score-threshold filtering (configurable, default 0.5)
- RAG escalation gate — skips retrieval for casual messages

### Phase 4 — Agents (complete)
- Full LangGraph graph with safety routing, crisis short-circuit, RAG integration
- GateAgent (safety + RAG classifiers), AlphaAgent (output validation)
- MemoryAgent with episodic store + working memory
- Semantic cache wired into graph (cache_check → cache_write)

### Phase 5 — Memory Architecture (complete)
- Two-tier working memory (current block + rolling backlog)
- Long-term gem collection (personal_fact, sensitive_disclosure, psych_pattern)
- Sensitive topic flag with topic-shift detection
- Personal info detection + gentle follow-up injection
- Special instructions as final pipeline stage (Stage 5)
- Session history sliding window (last 30 messages)
- Episodic context now wired into the system prompt (was a known gap)
- Semantic cache reformulation on hit (was verbatim repeat before)
- Psych profile removed from system prompt injection (stored only)

### Phase 6a — Turn Trace Store + Debug Dashboard (complete)
Every completed turn is now permanently recorded as a structured trace containing everything that happened: what the user said, which pipeline nodes fired and why, whether RAG was triggered and what was retrieved, which memory layers were injected, which model was called, how many tokens it used and what it cost, whether the quality check passed, and what the final response was — including the full assembled system prompt broken down by stage.

These traces are stored permanently in Redis and appended to a JSONL file that seeds future eval datasets. An interactive debug dashboard is served at `/eval`, letting you navigate sessions and turns via a sidebar, then inspect any turn across three views: a Chat Flow view showing message pairs with 7 toggleable pipeline detail sections (pipeline path, gate outputs, RAG chunks, LLM stats, memory context, alpha check, cache status); a Memory view showing all six memory layers live (session messages, working memory current/backlog, episodic store, long-term gems, semantic cache, sensitive flag); and a Prompt view showing the full assembled system prompt broken down stage by stage, each collapsible, with a copy button. All toggle states persist across page navigations. A terminal CLI (`view_traces.py`) provides the same data without the browser.

---

## Planned Features (Phase 6b+)
- **Phase 6b:** Eval datasets + fine-tuning data curation pipeline (uses JSONL traces from 6a)
- **Phase 6c:** LLM-as-judge eval pipeline, quality degradation alerts, user feedback collection
- **Guardrails (Phase 7):** Input filter (off-topic, harmful), output filter (expanded from AlphaAgent)
- **Fine-tuning (Phase 7):** Training data curation pipeline, fine-tuned vs base model benchmark
- **ChromaDB cleanup:** Sweep episodic store at global psych profile re-evaluation point; remove low-value turns
- **Recurring topic detection:** Episodic store signals when a topic appears across multiple sessions → new `recurring_topic` gem type
- **Nested/associative RAG:** One retrieved memory triggers a secondary retrieval pass — associative chains
- **Bot persona evolution:** Accumulated long-term gems drive dynamic identity template selection per user
- **Dynamic special instructions:** Psych profile drives system-injected override instructions automatically
- **Distrust/testing detection:** When semantic cache fires repeatedly, classify whether user is testing vs. genuinely re-asking

---

## Things That Will Trip You Up
- ChromaDB and pgvector have different query APIs — the `vector_store.py` abstraction handles this
- LangGraph state management is stateful — always pass full state, don't rely on implicit context
- OpenAI and Anthropic have different message format conventions — the provider abstraction normalizes this
- The sensitive flag persists in Redis across LangGraph invocations — each graph invocation is one turn
- `session_messages` passed to the agent is already windowed (last 30) — the full history is in `SessionStore`
- Psych patterns are written to ChromaDB but NEVER retrieved for prompt injection — stored only
- Episodic store has a score threshold of 0.60 — low-similarity chunks are filtered before injection
- Working memory rebuild fires at `turn_number % 20 == 0` — first rebuild happens at exactly turn 20
- The sensitive window slice uses `raised_at_turn - 1` as the index into the windowed session_messages list
- `TurnTrace.from_state()` is called with `{**state, "response": response}` — `state["response"]` is not yet set at node execution time (LangGraph merges the return dict after the node completes)
- Trace writes in `_crisis_node` and `_fallback_node` capture `routing_path` *before* the terminal label is merged — the label is inferrable from `safety_label=CRISIS` or `alpha_passed=False`
- The `/api/eval/memory` endpoint accesses `memory_agent._episodic_collection()` and `_longterm_collection()` directly (read-only) to dump all ChromaDB entries — not just top-k semantic search results
- Turn traces have no TTL in Redis — they are permanent. Use `ZREVRANGE turn_traces_sess:{session_id}` to list turns

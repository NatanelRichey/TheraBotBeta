# AI Wellness Companion — CLAUDE.md

## Project Overview
Production-grade AI conversational wellness companion. Demonstrates: multi-provider LLM integration, RAG, agents (LangGraph), prompt engineering, evaluation pipelines, guardrails, caching, and fine-tuning.

This is a portfolio project for a Generative AI Engineer Lead role. Every design decision should reflect production thinking.

## Tech Stack
- **Backend:** FastAPI + Python 3.12 (async throughout)
- **LLM Providers:** OpenAI, Anthropic (behind abstraction layer)
- **Agent Framework:** LangGraph
- **Vector DB:** ChromaDB (dev), pgvector (prod)
- **Cache:** Redis (semantic caching)
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
  core/                 → Config, logging, dependency injection
  models/               → Pydantic schemas (request/response models)
  services/             → All business logic lives here
    llm/                → Provider abstraction, routing, cost tracking
    prompts/            → Template management, pipeline, A/B testing
      templates.py      → Disk loader with in-memory cache
      pipeline.py       → Multi-stage system prompt assembler
      experiments.py    → ExperimentRunner — concurrent A/B variant execution
    rag/                → Embeddings, vector store, retrieval, grounding
    agents/             → LangGraph agents, memory, safety routing
    guardrails/         → Input/output filters, safety rules
    cache/              → Semantic cache, Redis client
    evaluation/         → LLM-as-judge, metrics, eval datasets
static/                 → Single-file chat UI (index.html) — served at GET /
scripts/                → Utility scripts (seeding, evals, fine-tuning)
data/
  prompts/              → Prompt template .txt files (version-controlled)
    identity_warm_v1.txt
    identity_warm_v2.txt
    identity_clinical_v1.txt
    format_short_v1.txt
    format_long_v1.txt
    safety_escalation_v1.txt
  chroma/               → ChromaDB vector store (gitignored)
  evals/                → Eval datasets in JSONL format (gitignored)
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
- **Prompts are data, not code.** Store prompt templates in `data/prompts/` as YAML or Jinja2 files. Load them dynamically.
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
- `data/chroma/` and `data/evals/` are gitignored (runtime data)

### Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Each phase gets its own feature branch
- Squash merge to main

## Current Phase
Update this as you progress:
**Active Phase: 2 — Prompt Engineering & Pipeline**

## RAG Architecture (Phase 3 + 4)

Two separate vector stores with different purposes:

```
knowledge_store  → static wellness content (DBT, psychoeducation, coping strategies)
                   shared across all users, seeded once, updated manually
                   lives in: data/knowledge/

episodic_store   → per-user conversation history
                   one collection per user_id, grows over time
                   embedded and retrieved by semantic similarity to current message
                   built in Phase 4 alongside MemoryAgent
```

Retrieval strategy:
- Phase 3: `knowledge_store` only
- Phase 4: both stores, results merged before context injection

Embedding model: TBD (decided in Phase 3)
Vector DB: ChromaDB (dev), pgvector (prod)

## MemoryAgent (Phase 4)

Manages two memory types:
- **Episodic memory:** past conversation turns embedded into `episodic_store`, retrieved by semantic similarity to current message
- **Working memory:** rolling summary of current session injected at the memory pipeline stage (the `# TODO` placeholder in `pipeline.py`)

The `# TODO` in `pipeline.py` stage 2 will be replaced by MemoryAgent output in Phase 4.

## Phase Notes
- **Phase 4 (Agents):** Wire semantic cache into the agent router — check cache before any LLM dispatch, write through on completion. Goal is to avoid redundant LLM calls for semantically equivalent inputs, which compounds across multi-step agent workflows.

## Key Commands
```bash
# Run the app
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run linter
ruff check app/ tests/

# Run type checker
mypy app/

# Run eval suite (slow)
pytest tests/evals/ -v -m eval

# Seed vector database (upsert — safe to re-run)
python scripts/seed_knowledge.py

# Seed vector database (reset collection first)
python scripts/seed_knowledge.py --reset

# Clean raw JSONL knowledge chunks before seeding
python scripts/clean_knowledge.py

# Run evaluation benchmark
python scripts/run_evals.py
```

## LLM Provider Notes
- OpenAI: Used for GPT-4o (primary conversation), text-embedding-3-small (embeddings)
- Anthropic: Used for Claude Sonnet (fallback conversation, safety checks)
- DeepSeek: OpenAI-compatible — reuses `OpenAIProvider` with `base_url="https://api.deepseek.com"`
- Kimi (Moonshot): OpenAI-compatible — reuses `OpenAIProvider` with `base_url="https://api.moonshot.cn/v1"`
- OpenRouter: OpenAI-compatible aggregator (`base_url="https://openrouter.ai/api/v1"`) — one key gives access to all models. When `OPENROUTER_API_KEY` is set, both routing profiles use OpenRouter instead of direct provider keys. No new provider class needed.
- Provider abstraction in `app/services/llm/base.py` — all providers implement `BaseLLMProvider`
- Router in `app/services/llm/router.py` handles model selection + automatic fallback

### Routing profiles
| Profile | Direct keys | Via OpenRouter |
|---------|-------------|----------------|
| `default` | `gpt-4o` → `claude-sonnet-4-6` | `openai/gpt-4o` → `anthropic/claude-sonnet-4-5` |
| `cheap` | `deepseek-chat` → `moonshot-v1-8k` | `deepseek/deepseek-chat` → `moonshot/moonshot-v1-8k` |

Profile is set per-request via `"profile": "default"` or `"profile": "cheap"` in the request body.
OpenRouter takes precedence when `OPENROUTER_API_KEY` is set — direct keys (`OPENAI_API_KEY`, etc.) are ignored for routing in that case.

## Phase Notes

### Phase 2 — Prompt Engineering & Pipeline (complete)
- `data/prompts/` — 6 .txt template files (identity ×3, format ×2, safety ×1)
- `templates.py` — disk loader, double-checked in-memory cache, structured log on first load
- `pipeline.py` — 4-stage assembler (identity → memory placeholder → safety → format)
- `experiments.py` — `ExperimentRunner`: 4 experiment types, `asyncio.gather` concurrency
- `Session.turn_count` — incremented on every user message; experiment fires on every 10th turn
- `POST /chat/compare` — explicit side-by-side comparison, returns both variants
- `POST /chat/compare/vote` — logs `comparison_vote` structured entry; no DB storage

**Stretch goal (Phase 2, not yet done):** Streaming compare UI in `static/index.html` — compare mode toggle, side-by-side rendering, Prefer button posting to `/chat/compare/vote`.

## Planned Features

## Things That Will Trip You Up
- ChromaDB and pgvector have different query APIs — the `vector_store.py` abstraction handles this
- LangGraph state management is stateful — always pass full state, don't rely on implicit context
- OpenAI and Anthropic have different message format conventions — the provider abstraction normalizes this
- Redis semantic cache uses cosine similarity — threshold is configurable in settings
- Evaluation datasets in `data/evals/` use JSONL format — one example per line

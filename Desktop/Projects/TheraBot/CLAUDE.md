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
  core/                 → Config, logging, dependency injection
  models/               → Pydantic schemas (request/response models)
  services/             → All business logic lives here
    llm/                → Provider abstraction, routing, cost tracking
    prompts/            → Template management, pipeline, A/B testing
    rag/                → Embeddings, vector store, retrieval, grounding
    agents/             → LangGraph agents, memory, safety routing
    guardrails/         → Input/output filters, safety rules
    cache/              → Semantic cache, Redis client
    evaluation/         → LLM-as-judge, metrics, eval datasets
scripts/                → Utility scripts (seeding, evals, fine-tuning)
data/                   → Knowledge base content, eval datasets, prompt files
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
- The `/data` directory is gitignored

### Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Each phase gets its own feature branch
- Squash merge to main

## Current Phase
Update this as you progress:
**Active Phase: 1 — Foundation**

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

# Seed vector database
python scripts/seed_knowledge.py

# Run evaluation benchmark
python scripts/run_evals.py
```

## LLM Provider Notes
- OpenAI: Used for GPT-4o (primary conversation), text-embedding-3-small (embeddings)
- Anthropic: Used for Claude Sonnet (fallback conversation, safety checks)
- Provider abstraction in `app/services/llm/base.py` — all providers implement `BaseLLMProvider`
- Router in `app/services/llm/router.py` handles model selection + automatic fallback

## Things That Will Trip You Up
- ChromaDB and pgvector have different query APIs — the `vector_store.py` abstraction handles this
- LangGraph state management is stateful — always pass full state, don't rely on implicit context
- OpenAI and Anthropic have different message format conventions — the provider abstraction normalizes this
- Redis semantic cache uses cosine similarity — threshold is configurable in settings
- Evaluation datasets in `data/evals/` use JSONL format — one example per line

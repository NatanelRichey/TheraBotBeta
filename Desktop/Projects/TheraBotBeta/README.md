# AI Wellness Companion

A production-grade AI conversational wellness system demonstrating: multi-provider LLM integration, RAG, LangGraph agents, prompt engineering pipelines, evaluation frameworks, guardrails, semantic caching, and fine-tuning.

## Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- API keys: OpenAI + Anthropic

### Setup

```bash
# Clone and enter the project
cd ai-wellness-companion

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Option A: Run with Docker
docker-compose up -d

# Option B: Run locally
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

### Verify
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Project Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation — FastAPI + LLM providers | 🔧 In Progress |
| 2 | Prompt Engineering & Pipeline | ⏳ Planned |
| 3 | RAG & Retrieval | ⏳ Planned |
| 4 | Agents & Multi-Step Workflows | ⏳ Planned |
| 5 | Cost, Latency & Reliability | ⏳ Planned |
| 6 | Evaluation, Monitoring & Guardrails | ⏳ Planned |
| 7 | Fine-Tuning & Model Customization | ⏳ Planned |

## Architecture

```
User → FastAPI → Guardrails (input) → Agent Router → LLM Provider
                                          ↓
                                    RAG Retrieval (if needed)
                                          ↓
                                    LLM Generation
                                          ↓
                                   Guardrails (output) → Response
```

## Development

```bash
# Tests
pytest tests/ -v

# Linting
ruff check app/ tests/

# Type checking
mypy app/

# Coverage
pytest tests/ --cov=app --cov-report=term-missing
```

## License
MIT

# TheraBot Deployment & Hosting Guide

## Architecture Decision: Custom Agentic Interface

**Recommendation: Build your own agentic interface using LangGraph + OpenAI APIs**

### Why Custom vs OpenAI Native Interfaces

#### ✅ **Your Custom Interface (Recommended)**
- **Full control** over agent orchestration (Safety → Memory → Therapist → Research → Reflection)
- **Specialized RAG** with metadata filtering, re-ranking, and domain-specific retrieval
- **Custom state management** with ConversationState, session persistence, and turn ordering
- **Observability** with TruLens/DeepEval hooks at each agent node
- **Flexibility** to swap models, add new agents, or modify flows without API constraints

#### ❌ **OpenAI Assistants API (Not Suitable)**
- **Single agent** design - can't orchestrate multiple specialized agents
- **Limited RAG control** - can't do metadata filtering, re-ranking, or custom retrieval logic
- **No custom state management** - can't enforce your ConversationState structure
- **Limited observability** - harder to instrument evaluation hooks per agent
- **Less flexible** - locked into OpenAI's assistant paradigm

#### ⚠️ **OpenAI Function Calling (Partial Use)**
- **Good for**: Individual agents (TherapistAgent, SafetyAgent) making OpenAI API calls with tools
- **Not for**: Overall orchestration - you still need LangGraph to coordinate agents

## Deployment Strategy

### Option 1: FastAPI Microservice (Recommended for Production)

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │ FastAPI │  ← Main API Gateway
    │ Service │
    └────┬────┘
         │
    ┌────┴──────────────────────────┐
    │ LangGraph Orchestrator        │
    │ - SessionManager              │
    │ - ConversationState           │
    │ - Agent Graph (Safety→...→Reflection)
    └────┬──────────────────────────┘
         │
    ┌────┴──────────────────────────────────┐
    │ External Services                      │
    │ - OpenAI APIs (GPT-4, Embeddings)     │
    │ - Vector DB (Pinecone/Weaviate)       │
    │ - Redis (Caching)                     │
    │ - Postgres (Session State)            │
    └───────────────────────────────────────┘
```

#### Implementation Structure

```
therabot/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── routes/
│   │   ├── session.py       # /session/{id}/turn
│   │   └── health.py
│   └── middleware/
│       ├── auth.py
│       └── logging.py
├── agents/
│   ├── __init__.py
│   ├── therapist_agent.py
│   ├── memory_agent.py
│   ├── research_agent.py    # RAG interface
│   ├── safety_agent.py
│   └── reflection_agent.py
├── graph/
│   ├── __init__.py
│   ├── orchestrator.py      # LangGraph setup
│   └── state.py             # ConversationState
├── rag/
│   ├── __init__.py
│   ├── retriever.py         # Vector DB interface
│   ├── embeddings.py        # OpenAI embeddings
│   └── reranker.py
├── services/
│   ├── session_manager.py
│   ├── openai_client.py     # OpenAI API wrapper
│   └── storage.py           # Postgres/Redis
└── config/
    └── settings.py          # Environment config
```

### Option 2: Serverless (AWS Lambda / Azure Functions)

**Pros**: Auto-scaling, pay-per-use, managed infrastructure
**Cons**: Cold starts, limited execution time, harder to maintain WebSocket connections

**Use Case**: If you expect sporadic traffic and want minimal infrastructure management

### Option 3: Containerized Service (Docker + Cloud)

**Recommended Platforms**:
- **Azure Container Apps**: Auto-scaling, integrated with Azure services
- **AWS ECS/Fargate**: Serverless containers, good for multi-AZ
- **GKE (Google Kubernetes Engine)**: Full K8s control
- **Railway/Render**: Simpler deployment, good for MVP

## Hosting Components

### 1. API Service (FastAPI)
- **Hosting**: Cloud container service or VM
- **Scaling**: Horizontal (multiple instances behind load balancer)
- **Environment Variables**:
  - `OPENAI_API_KEY`
  - `VECTOR_DB_API_KEY`
  - `POSTGRES_URL`
  - `REDIS_URL`

### 2. Vector Database (RAG Storage)
- **Options**:
  - **Pinecone**: Managed, easy setup, good performance
  - **Weaviate Cloud**: Open-source option, hybrid search
  - **Azure Cognitive Search**: Integrated with Azure ecosystem
  - **Qdrant Cloud**: Self-hostable alternative

### 3. Session State Storage
- **PostgreSQL** (JSONB for ConversationState)
- **DynamoDB** (if going serverless/AWS)
- **Redis** (for short-term session cache)

### 4. Caching Layer
- **Redis**: Cache frequent RAG queries, session state

### 5. Monitoring & Logging
- **OpenTelemetry** → ELK/Splunk
- **TruLens** for evaluation dashboards
- **Application Insights** (Azure) or **CloudWatch** (AWS)

## Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Vector database populated with DBT/DSM knowledge base
- [ ] OpenAI API keys secured (use secrets management)
- [ ] Database migrations run
- [ ] Health check endpoints implemented
- [ ] Rate limiting configured
- [ ] Authentication/authorization setup

### Security
- [ ] API keys stored in secrets manager (Azure Key Vault, AWS Secrets Manager)
- [ ] HTTPS/TLS enabled
- [ ] PHI data encryption at rest and in transit
- [ ] Audit logging for SafetyAgent escalations
- [ ] Regional data residency compliance
- [ ] Input validation and sanitization

### Observability
- [ ] Structured logging (JSON format)
- [ ] Metrics collection (latency, error rates, agent usage)
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Alerting for high-risk escalations
- [ ] Evaluation dashboards (TruLens)

### Scalability
- [ ] Load testing performed
- [ ] Auto-scaling policies configured
- [ ] Database connection pooling
- [ ] Cache warming strategy for common queries
- [ ] Rate limiting per user/session

## Example FastAPI Deployment

### Minimal FastAPI Structure

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.routes import session, health
from services.openai_client import get_openai_client
import os

app = FastAPI(title="TheraBot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(session.router, prefix="/api/v1", tags=["session"])

@app.on_event("startup")
async def startup_event():
    # Initialize OpenAI client
    get_openai_client()
    # Initialize vector DB connections
    # Warm up caches
    pass
```

### Dockerfile Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (Local Development)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/therabot
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=therabot
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## Cost Considerations

### OpenAI API Costs
- **GPT-4**: ~$0.03-0.06 per 1K input tokens, ~$0.06-0.12 per 1K output tokens
- **Embeddings (text-embedding-3-large)**: ~$0.00013 per 1K tokens
- **Estimate**: ~$0.10-0.50 per therapeutic session turn (with RAG retrieval + multiple agents)

### Infrastructure Costs
- **FastAPI Service**: $20-200/month (depending on instance size and traffic)
- **Vector DB**: $70-500/month (Pinecone/Weaviate)
- **PostgreSQL**: $15-100/month (managed service)
- **Redis**: $15-50/month (managed cache)

### Optimization Tips
1. **Cache RAG results** in Redis for common queries
2. **Use GPT-3.5-turbo** for SafetyAgent (cheaper, fast classification)
3. **Batch embedding** requests when possible
4. **Implement request queuing** to smooth out traffic spikes
5. **Use streaming responses** to improve perceived latency

## Next Steps

1. **Start with FastAPI + Docker** for local development
2. **Deploy to cloud container service** (Azure Container Apps recommended)
3. **Set up monitoring** (OpenTelemetry + logging)
4. **Load test** with synthetic sessions
5. **Implement gradual rollout** with feature flags


<!-- TheraBot Intelligence Layer Architecture doc -->

# TheraBot Intelligence Layer Architecture

## 1. System Overview

- **LangGraph multi-agent topology**
  - `TherapistAgent`: primary dialogue manager enforcing therapy structure (check-ins → exploration → interventions → homework). Consumes conversation state, orchestrates tool calls, and produces user-facing responses.
  - `MemoryAgent`: summarizes each exchange into short-term working memory, curates longitudinal insights (values, triggers, progress indicators), and syncs with the long-term store between turns.
  - `ResearchAgent`: interfaces with the psycho-education RAG retriever to fetch DBT/DSM/ICD references, skill descriptions, and evidence-based suggestions.
  - `SafetyAgent`: continuously classifies intent/sentiment, escalates to crisis playbooks, or injects grounding responses.
  - `ReflectionAgent`: validates Therapist outputs for hallucinations, contradictions against memory, and policy violations before user delivery.
- **Shared components**
  - `ConversationState`: structured object (session metadata, turn history, slots for retrieved docs, memory snapshots, safety flags).
  - `SessionManager`: handles LangGraph run contexts, enforces turn order, manages retries/timeouts, and routes escalations.
  - `LangChain Tooling Layer`: prompt templates, retrievers (Pinecone/Weaviate), memory abstractions, and tool wrappers that each agent invokes from LangGraph nodes.
  - `EvaluationHooks`: TruLens/DeepEval callbacks instrumented on critical nodes (Therapist, Safety, Reflection) for regression dashboards.

## 2. Data & Knowledge Strategy

- **Primary corpora**
  - `DBT Manual`: authoritative source for skills, worksheets, and treatment stages. Prioritize cleaned text, skill cards, and protocol summaries.
  - `DSM-5-TR` / `ICD-11`: symptom criteria, differential diagnosis cues, crisis thresholds (ingest selected sections respecting licensing).
  - `Session Transcripts`: de-identified, annotated with therapeutic intents, interventions, and outcomes for style grounding.
- **Ingestion & storage flow**
  1. Convert PDFs to structured Markdown/JSON; redact PHI and normalize typography.
  2. Chunk with overlap (400–600 tokens, 15% overlap) while preserving heading context for DBT skills; use smaller chunks (250–300 tokens) for dense DSM criteria.
  3. Attach metadata (`source`, `section`, `skill_phase`, `severity_level`, `safety_flag`, `version`, `licensing`).
  4. Embed via `text-embedding-3-large` (OpenAI) or `Instructor-XL` for open-source deployments; store vectors in Pinecone (therapy namespace, pod-level filtering) or Weaviate (hybrid search).
- **Retrieval tuning**
  - Use hybrid query: semantic + keyword (ensure acronyms like “TIPP”, “FAST” resolve).
  - Implement metadata filters (e.g., `skill_phase=mindfulness`, `severity_level=crisis`) to support therapist prompts.
  - Add re-ranker (Cohere Rerank or bge-reranker) `top_k=8 → rerank → top_3` for concise evidence snippets.
  - Cache high-frequency lookups (e.g., core DBT skills) in Redis + TTL for latency reduction.

## 3. Agent Interaction Design

- **Input/Output contracts**
  - `TherapistAgent`: inputs conversation state + memory snapshot + retrieved passages; outputs structured response `{utterance, rationale, follow_up_action, homework?}`.
  - `MemoryAgent`: inputs latest turn transcript; outputs `{working_memory_update, long_term_write?, journaling_prompt?}`.
  - `ResearchAgent`: inputs therapy intent + questions; outputs `{citations, bullet_summary, skill_steps}`.
  - `SafetyAgent`: inputs latest user message + safety context; outputs `{risk_level, action, escalation_target?}` with guardrail messages when risk ≥ medium.
  - `ReflectionAgent`: inputs proposed therapist reply + supporting docs; outputs `{verdict, issues[], revised_reply?}`.
- **Flow (per user turn)**
  1. SessionManager ingests user message → updates ConversationState.
  2. SafetyAgent runs first; if `risk_level≥high`, short-circuit to crisis playbook, notify escalation endpoint, and log event.
  3. MemoryAgent drafts `working_memory` summary for context window efficiency.
  4. TherapistAgent plans reasoning steps (ReAct) and decides whether to invoke ResearchAgent.
  5. ResearchAgent retrieves DBT/DSM/ICD snippets, returns citations for Therapist reasoning chain.
  6. TherapistAgent composes response referencing citations, ensures alignment with user goals and stage (e.g., DBT Stage 2).
  7. ReflectionAgent evaluates reply (hallucination checks, contradiction detection, safety guardrails). If fail, either request Therapist revision or auto-correct with grounded response.
  8. Approved reply returned to user; MemoryAgent persists long-term insights; EvaluationHooks record metrics.

## 4. Infrastructure & Observability

- **Runtime**
  - Deploy LangGraph graph as a FastAPI/Express microservice exposing `/session/{id}/turn`.
  - Use Redis Streams or Kafka for async escalation events to human supervisors.
  - Maintain session state in Postgres (JSONB conversation_state + vector references) or DynamoDB for serverless.
  - Containerize agents with shared base image; orchestrate via Azure Container Apps / ECS / GKE with autoscaling on concurrent sessions.
- **Security & compliance**
  - Encrypt memory stores at rest (KMS-managed keys), enforce regional data residency, and include audit logs for all Safety escalations.
  - Implement PHI tagging pipeline to ensure transcripts entering RAG store are de-identified and versioned.
- **Observability & evaluation**
  - Logging: structured events (turn_id, agent, latency, citations_used, safety_score) to OpenTelemetry + ELK/Splunk.
  - Quality evaluation: TruLens for alignment + attribution, DeepEval for regression suites (empathy, actionability, safety). Schedule nightly eval jobs on curated transcripts.
  - Crisis drill harness: synthetic crisis prompts replayed weekly; track mean time to escalation and false negatives.

## 5. Extension Hooks

- **Emotion & sentiment**: plug-in classifier (DistilRoBERTa, Microsoft GODEL emotion head) feeding emotion labels into Therapist planning and Memory insights.
- **Personalized plans**: planner module referencing Memory schema to auto-generate weekly goals, skill homework, and progress dashboards.
- **Chat history summarization**: periodic long-horizon summarizer agent writing thematic narratives (e.g., “Emotion regulation progress Q1”).
- **Audio modality**: Whisper-based ASR feeding transcripts into the same pipeline; optional TTS for Therapist responses using Azure Neural Voices.
- **Model fine-tuning readiness**: architecture keeps prompt templates modular and logs reasoning traces to enable future LoRA/QLoRA adaptation without refactoring core graph.


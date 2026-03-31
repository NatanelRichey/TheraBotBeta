# TheraBotBeta — Architecture Reference

This document consolidates the graph topology, prompt injection stages, and memory layer documentation in one place. Replaces `GRAPH_TOPOLOGY.txt`, `PROMPT_INJECTION_README.txt`, and `MEMORY_MILESTONES_README.txt`.

---

## LangGraph Graph Topology

```
START
  └─► cache_check ── HIT ──► reformulate (cheap LLM) ──────────────► END
              │
             MISS
              ▼
        gate_and_memory  ← 7-way asyncio.gather fan-out
          ├── safety classification       → SAFE | REDIRECT | CRISIS
          ├── RAG gate                   → bool (skip ChromaDB for casual messages)
          ├── sensitive topic detection  → bool (trauma / intimate disclosure)
          ├── personal info detection    → bool (family, location, health, etc.)
          ├── sensitive flag check       → load Redis flag, run topic-shift if active
          ├── two-tier working memory    → load / rebuild wm_current + wm_backlog
          ├── episodic retrieval         → top-k similar past turns from ChromaDB
          └── long-term gem retrieval    → personal_fact + sensitive_disclosure from ChromaDB
              │
        route_safety
          ├── CRISIS  → crisis_node ──────────────────────────────────► END
          ├── SAFE / REDIRECT + rag_required  → rag_retrieve
          └── SAFE / REDIRECT, no rag         → llm_dispatch
              │
        rag_retrieve  → llm_dispatch
              │
        llm_dispatch  (builds 6-stage system prompt, calls dispatch router)
              │
        alpha_check
          ├── PASS → cache_write ─────────────────────────────────────► END
          └── FAIL → fallback_node ──────────────────────────────────► END
```

### cache_write fires (fire-and-forget background tasks):
- Semantic cache write
- Episodic write (with richer metadata)
- Personal fact extraction LLM call (if `personal_info_detected`)
- Sensitive disclosure summary LLM call (if `sensitive_flag_dropping`)
- Working memory rebuild + psych pattern extraction (if `turn_number % 20 == 0`)
- **Turn trace write** (Phase 6a) — fired from all 4 terminal nodes (cache hit, cache_write, crisis, fallback)

### Turn Trace (Phase 6a)
Every completed turn is persisted as a `TurnTrace` record:
- **What's captured:** user message, full routing path, gate outputs (safety label/confidence, RAG, sensitive, personal_info), RAG results (chunks + scores), memory context injected, LLM model + tokens + cost, alpha check result, full assembled system prompt, final response
- **Storage:** Redis `turn_trace:{session_id}:{turn_number}` (permanent) + JSONL `data/evals/traces.jsonl` (append-only)
- **Indexes:** `turn_traces_sess:{session_id}` and `turn_traces:{user_id}` — sorted sets, score = epoch timestamp
- **Access:** GET `/api/eval/*` endpoints or `scripts/view_traces.py` CLI
- **Dashboard:** Interactive UI at `/eval` (Chat Flow / Memory / Prompt tabs, all sections toggleable)

---

## System Prompt Pipeline (6 Stages)

Every turn that is not a cache hit assembles a system prompt from these ordered stages:

| Stage | Content | File | Condition |
|---|---|---|---|
| **1** | Identity template (warm / clinical persona) | `data/prompts/identity_*.txt` | Always |
| **2** | DBT knowledge context (retrieved skill handouts) | `grounding.format_context()` | `rag_required=True` |
| **2.5** | Session memory (two-tier working memory) | `pipeline.py` | When memory exists |
| **2.5** | Relevant past conversations (episodic chunks) | `grounding.format_episodic_context()` | Score ≥ 0.60 |
| **2.5** | What I know about you (personal facts + disclosures) | `grounding.format_longterm_context()` | Score ≥ 0.55 |
| **3** | Safety escalation block | `data/prompts/safety_escalation_v1.txt` | `safety_label=REDIRECT` |
| **4** | Format template (short / long) | `data/prompts/format_*.txt` | Always |
| **5** | **Special instructions** | `pipeline.py` | Any active override |

**Stage 5 is injected last.** The model reads it with highest recency weight. The header explicitly states these instructions take precedence over all other guidelines.

### Current Stage 5 triggers:
- `personal_info_detected=True` → "Gently explore further with one warm follow-up question"
- `sensitive_flag_active=True` → "SENSITIVE TOPIC ACTIVE: listen fully, do not probe or redirect"

### What is NOT injected:
- `psych_pattern` entries — stored in `lt_{user_id}` ChromaDB but never injected. Reserved for future persona evolution.

---

## Memory Layer Reference

| Layer | Key / Collection | Technology | TTL | Scope |
|---|---|---|---|---|
| Semantic cache | `semantic_cache:{session_id}` | Redis list | 1h | Session |
| Raw session history | In-memory dict | SessionStore | Runtime | Session |
| Working memory — current | `wm_current:{session_id}` | Redis string | 24h | Session |
| Working memory — backlog | `wm_backlog:{session_id}` | Redis string | 24h | Session |
| Sensitive topic flag | `sensitive_flag:{session_id}` | Redis string | Until dropped | Session |
| RAG trigger accumulator | `psych_triggers:{user_id}` | Redis list | 7 days | User |
| Episodic turns | `ep_{user_id}` | ChromaDB `data/episodic/` | Permanent | User |
| Long-term gems | `lt_{user_id}` | ChromaDB `data/longterm/` | Permanent | User |
| DBT knowledge | `wellness_knowledge` | ChromaDB `data/chroma/` | Permanent | Shared |

### Long-term gem types (`lt_{user_id}`):

| `memory_type` | Trigger | Injected? | Build path |
|---|---|---|---|
| `personal_fact` | `classify_personal_info()` → background LLM extraction | Yes, via similarity | Per detected message |
| `sensitive_disclosure` | Sensitive flag drops → LLM summarizes window | Yes, via similarity | Per sensitive episode |
| `psych_pattern` | Every 20-turn rebuild (separate LLM call) + RAG threshold | **No** | Session rebuild + RAG accumulation |

---

## GateAgent Classifiers

All 5 run concurrently via `asyncio.gather`. Each uses the cheap router with `max_tokens=5`, `temperature=0.0`.

| Method | Returns | Used when |
|---|---|---|
| `classify_safety()` | `SAFE / REDIRECT / CRISIS` | Every non-cached turn |
| `classify_rag()` | `bool` | Every non-cached turn |
| `classify_sensitive()` | `bool` | Every non-cached turn |
| `classify_personal_info()` | `bool` | Every non-cached turn |
| `classify_topic_shift()` | `bool` | Only when sensitive flag is already active |

---

## AgentState Fields

```python
# Request
session_id, user_id, user_message, turn_number, session_messages (windowed)

# Gate outputs
safety_label, safety_confidence, rag_required

# Sensitive flag
sensitive_flag_active, sensitive_flag_dropping, sensitive_flag_raised_at_turn

# Personal info
personal_info_detected

# Memory outputs
working_memory, episodic_chunks, longterm_gems, rag_trigger_count

# Special instructions (Stage 5)
special_instructions: list[str]

# RAG
retrieved_knowledge, rewritten_query

# LLM dispatch
raw_response

# Alpha validation
alpha_passed, alpha_flags (append-only)

# Final
response, routing_path (append-only audit trail), cache_hit
```

---

## Two-Tier Working Memory

```
Turn 0-20:    wm_current = summary of turns 1-20
              wm_backlog = empty

Turn 21-40:   old wm_current → LLM merge → wm_backlog
              wm_current = summary of turns 21-40

Turn 41-60:   old wm_current → merged into existing wm_backlog
              wm_current = summary of turns 41-60

Turn N+:      wm_backlog stays bounded (merging compressed-into-compressed)
              wm_current always represents the latest 20-turn block
```

Injected as a single `## Session Memory` block:
```
[Session History]
{wm_backlog}

[Recent]
{wm_current}
```

---

## Sensitive Topic Lifecycle

```
Turn 12: classify_sensitive() → True + flag not active
         → set_sensitive_flag(session_id, raised_at_turn=12) in Redis
         → special_instructions gets listening instruction

Turn 13-15: classify_sensitive() → True + flag active
            → sensitive_flag_active=True, stay in listening mode

Turn 16: classify_sensitive() → False + flag active
         → classify_topic_shift() → True
         → sensitive_flag_dropping=True
         → clear_sensitive_flag() (fire-and-forget)
         → cache_write: write_sensitive_disclosure(window=messages[11:]) (fire-and-forget)
         → sensitive_disclosure gem written to lt_{user_id} in ChromaDB
```

---

## Key Config Values

```python
working_memory_rebuild_interval = 20   # turns between rebuilds
session_history_window = 30            # max messages passed to LLM per turn
semantic_cache_threshold = 0.92        # cosine similarity for cache hit (1h TTL)
episodic_top_k = 3                     # past turns retrieved per query
psych_profile_rag_threshold = 5        # RAG triggers before psych pattern build
longterm_chroma_dir = "./data/longterm"
episodic_chroma_dir = "./data/episodic"
```

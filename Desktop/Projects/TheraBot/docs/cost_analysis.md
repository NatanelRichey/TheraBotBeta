# TheraBot Cost Analysis: 15-Minute Demo Session

## Session Assumptions

- **Duration**: 15 minutes (900 seconds)
- **Average turn time**: ~45 seconds (user thinks + bot processes + bot responds)
- **Estimated turns**: ~20 turns per session
- **Conversation style**: Therapeutic dialogue (not rapid-fire chat)

## Cost Breakdown Per Turn

### 1. SafetyAgent (Runs First - Every Turn)
**Model**: GPT-3.5-turbo (cheaper, fast classification)
- **Input tokens**: ~500 (user message + safety context)
- **Output tokens**: ~50 (risk classification + action)
- **Cost**: 
  - Input: 500 × $0.0005/1K = **$0.00025**
  - Output: 50 × $0.0015/1K = **$0.000075**
  - **Total: $0.000325 per turn**

### 2. MemoryAgent (Every Turn)
**Model**: GPT-4-turbo
- **Input tokens**: ~1,000 (conversation history + working memory)
- **Output tokens**: ~200 (memory update summary)
- **Cost**:
  - Input: 1,000 × $0.01/1K = **$0.01**
  - Output: 200 × $0.03/1K = **$0.006**
  - **Total: $0.016 per turn**

### 3. TherapistAgent - Planning (Every Turn)
**Model**: GPT-4-turbo with Function Calling
- **Input tokens**: ~2,000 (conversation state + memory + therapy context)
- **Output tokens**: ~100 (planning decision + function call if needed)
- **Cost**:
  - Input: 2,000 × $0.01/1K = **$0.02**
  - Output: 100 × $0.03/1K = **$0.003**
  - **Total: $0.023 per turn**

### 4. ResearchAgent (Runs ~60% of turns - when RAG needed)
**Model**: text-embedding-3-large
- **Query tokens**: ~100 (therapy question)
- **Embedding cost**: 100 × $0.00013/1K = **$0.000013**
- **Vector DB query**: Pinecone/Weaviate - **$0.0001** (very cheap)
- **Re-ranking**: Cohere Rerank - **$0.0001** (if used)
- **Total: $0.000213 per RAG call**
- **Per turn (60% probability)**: $0.000213 × 0.6 = **$0.000128**

### 5. TherapistAgent - Response Composition (Every Turn)
**Model**: GPT-4-turbo
- **Input tokens**: ~2,500 (conversation + memory + retrieved docs + context)
- **Output tokens**: ~500 (therapeutic response)
- **Cost**:
  - Input: 2,500 × $0.01/1K = **$0.025**
  - Output: 500 × $0.03/1K = **$0.015**
  - **Total: $0.04 per turn**

### 6. ReflectionAgent (Every Turn)
**Model**: GPT-4-turbo
- **Input tokens**: ~2,500 (proposed response + retrieved docs + memory)
- **Output tokens**: ~100 (validation verdict)
- **Cost**:
  - Input: 2,500 × $0.01/1K = **$0.025**
  - Output: 100 × $0.03/1K = **$0.003**
  - **Total: $0.028 per turn**

## Total Cost Per Turn

| Agent | Cost per Turn |
|-------|---------------|
| SafetyAgent | $0.000325 |
| MemoryAgent | $0.016 |
| TherapistAgent (Planning) | $0.023 |
| ResearchAgent (RAG) | $0.000128 |
| TherapistAgent (Compose) | $0.04 |
| ReflectionAgent | $0.028 |
| **Total per Turn** | **$0.107** |

*Note: Some turns may have ReflectionAgent trigger a revision, adding ~$0.028 per revision (rare, ~10% of turns)*

## 15-Minute Demo Session Cost

### API Costs (20 turns)
- **Base cost**: 20 turns × $0.107 = **$2.14**
- **Revisions** (10% of turns): 2 × $0.028 = **$0.056**
- **Total API cost**: **~$2.20**

### Infrastructure Costs (Prorated for Demo)

For a **single demo session**, infrastructure costs are negligible if you're using:
- **Free tier services** (e.g., Railway, Render free tier)
- **Pay-as-you-go** (only pay for what you use)

However, if running on dedicated infrastructure:

| Service | Monthly Cost | Per 15-min Session |
|---------|-------------|-------------------|
| FastAPI Hosting (small instance) | $20 | $0.001 |
| Vector DB (Pinecone Starter) | $70 | $0.003 |
| PostgreSQL (small) | $15 | $0.001 |
| Redis Cache | $15 | $0.001 |
| **Total Infrastructure** | **$120/month** | **~$0.006** |

*Infrastructure cost per session is tiny because it's shared across many sessions*

## Total Cost for 15-Minute Demo

### Scenario 1: Using Free/Shared Infrastructure
- **API costs**: $2.20
- **Infrastructure**: $0 (free tier or shared)
- **Total**: **~$2.20**

### Scenario 2: Using Dedicated Infrastructure
- **API costs**: $2.20
- **Infrastructure (prorated)**: $0.006
- **Total**: **~$2.21**

## Cost Optimization Strategies

### For Demo/Development

1. **Use GPT-3.5-turbo for More Agents**
   - SafetyAgent: ✅ Already using GPT-3.5-turbo
   - MemoryAgent: Could use GPT-3.5-turbo (saves ~$0.014/turn)
   - ReflectionAgent: Could use GPT-3.5-turbo (saves ~$0.025/turn)
   - **Savings**: ~$0.78 per 15-min session

2. **Reduce Context Window**
   - Limit conversation history to last 3-5 turns
   - **Savings**: ~$0.30 per 15-min session

3. **Cache RAG Results**
   - Cache common DBT skill queries in Redis
   - **Savings**: ~$0.01 per 15-min session

4. **Skip ReflectionAgent for Demo**
   - Only use in production
   - **Savings**: ~$0.56 per 15-min session

### Optimized Demo Cost

If you implement optimizations 1, 2, and 4:
- **Original**: $2.20
- **Optimized**: **~$0.56 per 15-min session**

## Cost Comparison: Different Scenarios

| Scenario | Cost per 15-min Session |
|----------|------------------------|
| **Full Production Setup** (all GPT-4) | $2.20 |
| **Optimized Demo** (GPT-3.5 where possible) | $0.56 |
| **Minimal Demo** (skip ReflectionAgent) | $0.40 |
| **Ultra-Cheap Demo** (GPT-3.5 + minimal context) | $0.25 |

## Monthly Cost Estimates

### Development/Demo Usage
- **10 demo sessions/day** = 300 sessions/month
- **Cost**: 300 × $0.56 = **$168/month** (optimized)
- **Cost**: 300 × $2.20 = **$660/month** (full production)

### Production Usage
- **100 sessions/day** = 3,000 sessions/month
- **Cost**: 3,000 × $2.20 = **$6,600/month** (API only)
- **Infrastructure**: $120/month
- **Total**: **~$6,720/month**

## Cost Breakdown Visualization

### Per 15-Minute Session (Full Production)
```
┌─────────────────────────────────────┐
│ Total: $2.20                        │
├─────────────────────────────────────┤
│ ReflectionAgent:     $0.56 (25%)   │
│ TherapistAgent:      $1.26 (57%)   │
│ MemoryAgent:         $0.32 (15%)   │
│ SafetyAgent:         $0.01 (0.5%)  │
│ ResearchAgent:       $0.003 (0.1%) │
│ Revisions:           $0.06 (2.7%)  │
└─────────────────────────────────────┘
```

### Key Insights

1. **TherapistAgent is the biggest cost** (57%) - uses GPT-4 for both planning and response
2. **ReflectionAgent is second** (25%) - but critical for safety/quality
3. **RAG is very cheap** (<0.1%) - embeddings are inexpensive
4. **SafetyAgent is negligible** - using GPT-3.5-turbo keeps it cheap

## Recommendations for Demo

### Option 1: Full Fidelity Demo ($2.20)
- Use all agents with GPT-4
- Best quality, realistic production experience
- Good for investor demos, user testing

### Option 2: Optimized Demo ($0.56)
- Use GPT-3.5-turbo for MemoryAgent and ReflectionAgent
- Still high quality, 75% cost savings
- Good for development, internal testing

### Option 3: Minimal Demo ($0.25)
- GPT-3.5-turbo for most agents
- Skip ReflectionAgent
- Reduced context window
- Good for early prototyping

## Additional Cost Considerations

### One-Time Setup Costs
- **Vector DB indexing**: Free (one-time embedding of knowledge base)
- **Infrastructure setup**: $0 (if using free tiers)

### Variable Costs
- **Storage**: Negligible for demo (<$1/month)
- **Bandwidth**: Negligible for demo (<$1/month)
- **Monitoring**: Free tier available (OpenTelemetry → free logging)

## Real-World Example: 15-Minute Session

**Turn 1**: User: "I'm feeling really anxious today"
- SafetyAgent: $0.0003
- MemoryAgent: $0.016
- TherapistAgent (plan): $0.023
- ResearchAgent (RAG): $0.0001 (anxiety → DBT skills)
- TherapistAgent (compose): $0.04
- ReflectionAgent: $0.028
- **Turn 1 Total: $0.107**

**Turn 2**: User: "What can I do right now?"
- SafetyAgent: $0.0003
- MemoryAgent: $0.016
- TherapistAgent (plan): $0.023
- ResearchAgent (RAG): $0.0001 (distress tolerance skills)
- TherapistAgent (compose): $0.04
- ReflectionAgent: $0.028
- **Turn 2 Total: $0.107**

... (18 more turns)

**Session Total**: 20 × $0.107 = **$2.14** + minor revisions = **~$2.20**

## Summary

**For a 15-minute demo session:**
- **Full production setup**: **$2.20**
- **Optimized demo**: **$0.56**
- **Minimal demo**: **$0.25**

**Recommendation**: Start with **optimized demo ($0.56)** for development, then scale to full production ($2.20) for important demos or production use.



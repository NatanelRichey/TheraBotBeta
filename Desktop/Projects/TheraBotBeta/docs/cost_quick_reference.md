# Cost Quick Reference: 15-Minute Demo Session

## TL;DR

**15-minute demo session cost:**
- **Full production**: **$2.20**
- **Optimized demo**: **$0.56** (recommended for demos)
- **Minimal demo**: **$0.25** (for early prototyping)

## Quick Breakdown

### Full Production Setup ($2.20 per 15-min session)
- ~20 turns in 15 minutes
- All agents use GPT-4 (except SafetyAgent uses GPT-3.5-turbo)
- Full RAG retrieval
- ReflectionAgent validation

### Optimized Demo ($0.56 per 15-min session)
- Same ~20 turns
- GPT-3.5-turbo for MemoryAgent and ReflectionAgent
- Full RAG retrieval
- **75% cost savings** with minimal quality loss

### Minimal Demo ($0.25 per 15-min session)
- GPT-3.5-turbo for most agents
- Skip ReflectionAgent
- Reduced context window
- **88% cost savings** for early testing

## Cost Per Component

| Component | Cost per Turn | Cost per 15-min Session |
|-----------|---------------|-------------------------|
| SafetyAgent (GPT-3.5) | $0.0003 | $0.006 |
| MemoryAgent (GPT-4) | $0.016 | $0.32 |
| TherapistAgent (GPT-4) | $0.063 | $1.26 |
| ResearchAgent (RAG) | $0.0001 | $0.002 |
| ReflectionAgent (GPT-4) | $0.028 | $0.56 |
| **Total** | **$0.107** | **$2.20** |

## Monthly Estimates

| Usage | Full Production | Optimized Demo |
|-------|----------------|----------------|
| 10 sessions/day (300/month) | $660 | $168 |
| 50 sessions/day (1,500/month) | $3,300 | $840 |
| 100 sessions/day (3,000/month) | $6,600 | $1,680 |

## Cost Optimization Tips

1. **Use GPT-3.5-turbo for non-critical agents** → Save 75%
2. **Cache RAG results** → Save ~$0.01 per session
3. **Limit conversation history** → Save ~$0.30 per session
4. **Skip ReflectionAgent for demos** → Save $0.56 per session

## Infrastructure Costs

- **Free tier available** (Railway, Render) → $0 for demos
- **Dedicated infrastructure** → ~$120/month (prorated: $0.006 per session)

## Real Example

**15-minute therapeutic conversation:**
- User asks about anxiety → Bot retrieves DBT skills → Provides response
- User asks follow-up → Bot references memory → Provides personalized response
- ... (18 more exchanges)
- **Total: ~$2.20** (full) or **~$0.56** (optimized)

## Recommendation

**For demos**: Use **optimized setup ($0.56)** - still high quality, 75% cheaper
**For production**: Use **full setup ($2.20)** - maximum safety and quality



"""Compare query-rewrite retrieval quality across LLM models."""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.core.config import get_settings
from app.services.llm.cost_tracker import CostTracker
from app.services.llm.openai_provider import OpenAIProvider
from app.services.llm.router import LLMRouter
from app.services.rag.retriever import COLLECTION_NAME, EMBEDDING_MODEL, KnowledgeRetriever

OPENROUTER_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "DeepSeek":   "deepseek/deepseek-chat",
    "Kimi k2.5":  "moonshotai/kimi-k2.5",
    "GPT-4o":     "openai/gpt-4o",
    "Sonnet":     "anthropic/claude-sonnet-4-5",
}

MESSAGES = [
    ("anxiety",        "I feel so anxious I cant breathe"),
    ("sadness",        "I have been crying all day and I dont know why"),
    ("anger",          "I get so angry I say things I regret and I hate myself for it"),
    ("shame",          "I feel like such a failure, I am worthless and everyone would be better off without me"),
    ("guilt",          "I hurt someone I love and I cannot forgive myself"),
    ("crisis urge",    "I am about to do something really destructive right now"),
    ("self harm",      "The urge to hurt myself is really strong tonight"),
    ("cant calm",      "I am spiraling and I cannot get my body to calm down"),
    ("rage",           "I want to throw things and scream, I am so furious"),
    ("nightmare",      "I keep having the same terrifying nightmare and I dread going to sleep"),
    ("dear man",       "I need to ask my boss for a raise but I freeze up every time"),
    ("conflict",       "My partner and I fight about the same thing over and over"),
    ("people please",  "I can never say no to anyone even when I am exhausted"),
    ("loneliness",     "I feel completely invisible, like no one actually sees me"),
    ("toxic relation", "I know this friendship is bad for me but I cannot leave"),
    ("rumination",     "I replay the same embarrassing memory on a loop and cannot stop"),
    ("logic/emotion",  "I know logically everything is fine but I feel like something terrible is about to happen"),
    ("overthinking",   "I overanalyze every conversation I have and assume people hate me"),
    ("black/white",    "Everything feels either perfect or completely ruined, there is no middle ground"),
    ("catastrophise",  "One small mistake and I am convinced my entire life is falling apart"),
    ("sleep",          "My mind races all night and I have not slept properly in weeks"),
    ("no motivation",  "I cannot get off the couch, everything feels pointless and heavy"),
    ("alcohol",        "I drink to cope when things get bad and then feel worse"),
    ("dissociation",   "I feel completely disconnected from my body like I am watching myself from outside"),
    ("burnout",        "I am so burned out I feel nothing at all anymore"),
    ("vague",          "I just feel bad"),
    ("very long",      "My mother called me again today and said the same hurtful things she always says and I just sat there feeling small and then I got off the phone and cried for an hour and now I am angry at myself for even picking up"),
    ("non clinical",   "Can you recommend a good book"),
    ("already skilled","I tried opposite action but it did not work, what else can I do"),
    ("positive",       "Things are actually going well but I am scared it will all fall apart"),
]

SPOTLIGHT = [0, 7, 11, 18, 25, 27, 28, 29]  # interesting cases to show rewrites for


def make_router(api_key: str, model_id: str) -> tuple[LLMRouter, CostTracker]:
    tracker = CostTracker()
    provider = OpenAIProvider(api_key=api_key, base_url=OPENROUTER_URL)
    router = LLMRouter(
        primary=provider, fallback=provider,
        cost_tracker=tracker,
        primary_model=model_id, fallback_model=model_id,
    )
    return router, tracker


async def run_model(
    name: str,
    model_id: str,
    collection: chromadb.Collection,
    api_key: str,
) -> tuple[list[float], list[str], float, dict]:
    router, tracker = make_router(api_key, model_id)
    retriever = KnowledgeRetriever(collection=collection, router=router)
    scores: list[float] = []
    rewrites: list[str] = []
    for _label, msg in MESSAGES:
        try:
            rewritten = await retriever.rewrite_query(msg)
            raw = collection.query(
                query_texts=[rewritten], n_results=1, include=["distances"]
            )
            score = round(1.0 - raw["distances"][0][0], 4)
        except Exception as exc:
            rewritten = f"ERROR: {exc}"
            score = 0.0
        scores.append(score)
        rewrites.append(rewritten)
    return scores, rewrites, tracker.stats


async def main() -> None:
    settings = get_settings()

    ef = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key, model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

    all_results: dict[str, tuple[list[float], list[str], float]] = {}

    for name, model_id in MODELS.items():
        print(f"Running {name} ({model_id})...", flush=True)
        t0 = time.time()
        scores, rewrites, cost_stats = await run_model(
            name, model_id, collection, settings.openrouter_api_key
        )
        elapsed = time.time() - t0
        all_results[name] = (scores, rewrites, elapsed, cost_stats)
        avg = sum(scores) / len(scores)
        h55 = sum(1 for s in scores if s >= 0.55)
        h60 = sum(1 for s in scores if s >= 0.60)
        cost = cost_stats["total_cost_usd"]
        itok = cost_stats["total_input_tokens"]
        otok = cost_stats["total_output_tokens"]
        print(f"  {elapsed:.0f}s  avg={avg:.3f}  >=0.55: {h55}/30  >=0.60: {h60}/30  cost: ${cost:.5f}  tokens: {itok}in/{otok}out")

    names = list(MODELS.keys())
    COL = 13

    # --- Score table ---
    print()
    header = f"{'Label':<16}" + "".join(f"{n:>{COL}}" for n in names)
    print(header)
    print("-" * len(header))

    for i, (label, _) in enumerate(MESSAGES):
        row = [all_results[n][0][i] for n in names]
        best = max(row)
        line = f"{label:<16}"
        for j, n in enumerate(names):
            s = row[j]
            cell = f"{s:.3f}{'*' if s == best else ' '}"
            line += f"{cell:>{COL}}"
        print(line)

    print("-" * len(header))

    def stat_row(label: str, fn) -> None:
        print(f"{label:<16}" + "".join(f"{fn(all_results[n][0]):>{COL}}" for n in names))

    stat_row("AVERAGE",    lambda s: f"{sum(s)/len(s):.3f}")
    stat_row("HITS >=0.55", lambda s: f"{sum(1 for x in s if x>=0.55)}/30")
    stat_row("HITS >=0.60", lambda s: f"{sum(1 for x in s if x>=0.60)}/30")
    stat_row("BEST on",     lambda s: f"{sum(1 for i,x in enumerate(s) if x==max(all_results[n][0][i] for n in names))}")
    print(f"{'TIME (s)':<16}" + "".join(f"{all_results[n][2]:.0f}s{'':{COL-3}}" for n in names))
    print(f"{'COST (USD)':<16}" + "".join(f"${all_results[n][3]['total_cost_usd']:.5f}{'':{COL-8}}" for n in names))
    print(f"{'$/query':<16}" + "".join(f"${all_results[n][3]['total_cost_usd']/30*1000:.4f}m{'':{COL-8}}" for n in names))
    print(f"{'input tok':<16}" + "".join(f"{all_results[n][3]['total_input_tokens']:>{COL}}" for n in names))
    print(f"{'output tok':<16}" + "".join(f"{all_results[n][3]['total_output_tokens']:>{COL}}" for n in names))

    # --- Rewrite spotlight ---
    print("\n=== REWRITE SAMPLES ===")
    for i in SPOTLIGHT:
        label, msg = MESSAGES[i]
        print(f'\n[{label}] "{msg[:70]}"')
        for n in names:
            rw = all_results[n][1][i]
            sc = all_results[n][0][i]
            print(f"  {n:<13} {sc:.3f}  {rw[:72]}")


if __name__ == "__main__":
    asyncio.run(main())

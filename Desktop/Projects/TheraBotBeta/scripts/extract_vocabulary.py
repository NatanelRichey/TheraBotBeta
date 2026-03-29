"""
extract_vocabulary.py — Extract unique modules and skill categories from the seeded collection.

Useful for inspecting what terminology the retrieval query-rewrite prompt can reference,
and for verifying the collection was seeded correctly.

Usage:
    python scripts/extract_vocabulary.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402

COLLECTION_NAME = "wellness_knowledge"


def main() -> None:
    settings = get_settings()
    openai_api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    ef = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

    # Always use local PersistentClient — no server needed
    persist_path = str(ROOT / settings.chroma_persist_dir.lstrip("./"))
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    modules = sorted(set(m["module"] for m in metadatas))
    skill_categories = sorted(set(m["skill_category"] for m in metadatas))
    handout_ids = sorted(set(m["handout_id"] for m in metadatas))
    content_types = sorted(set(m["content_type"] for m in metadatas))

    print(f"Total chunks: {len(metadatas)}\n")

    print(f"MODULES ({len(modules)}):")
    for m in modules:
        count = sum(1 for meta in metadatas if meta["module"] == m)
        print(f"  {m:<40} ({count} chunks)")

    print(f"\nCONTENT TYPES ({len(content_types)}):")
    for ct in content_types:
        count = sum(1 for meta in metadatas if meta["content_type"] == ct)
        print(f"  {ct:<40} ({count} chunks)")

    print(f"\nSKILL CATEGORIES ({len(skill_categories)}):")
    for s in skill_categories:
        print(f"  - {s}")

    print(f"\nHANDOUT IDs ({len(handout_ids)}):")
    for h in handout_ids:
        print(f"  - {h}")

    # Also print the vocabulary string as it will appear in the prompt
    print("\n" + "=" * 60)
    print("VOCABULARY STRING (as injected into RETRIEVAL_QUERY_REWRITE_PROMPT):")
    print("=" * 60)
    vocab = "\n".join(f"- {s}" for s in skill_categories)
    print(vocab)


if __name__ == "__main__":
    main()

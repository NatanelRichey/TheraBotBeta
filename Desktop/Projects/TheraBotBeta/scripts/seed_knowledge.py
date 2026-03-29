"""
seed_knowledge.py — Seed ChromaDB wellness_knowledge collection from cleaned JSONL files.

Usage:
    python scripts/seed_knowledge.py            # upsert (safe to re-run)
    python scripts/seed_knowledge.py --reset    # delete + recreate collection first

Source: data/knowledge/labeled/clean/*.jsonl
Target: ChromaDB collection  wellness_knowledge  (cosine, text-embedding-3-small)

Environment:
    CHROMA_MODE=local   Use PersistentClient at chroma_persist_dir (default, no server needed)
    CHROMA_MODE=http    Use HttpClient at chroma_url (requires ChromaDB server running)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so app.core.config is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLEAN_DIR = ROOT / "data" / "knowledge" / "labeled" / "clean"
COLLECTION_NAME = "wellness_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50
TEST_QUERY = "I feel so anxious and overwhelmed"
TOP_K = 3
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles on each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_id(value: str) -> str:
    """Lowercase, spaces → underscores, strip everything except [a-z0-9_]."""
    value = value.lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value


def _make_chunk_id(module: str, handout_id: str) -> str:
    """Stable ID derived from content identity, not file position."""
    return f"{_sanitize_id(module)}_{_sanitize_id(handout_id)}"


def _parse_chroma_host_port(url: str) -> tuple[str, int, bool]:
    """Return (host, port, ssl) from a URL like http://localhost:8001."""
    ssl = url.startswith("https://")
    url = url.removeprefix("https://").removeprefix("http://")
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        return host, int(port_str), ssl
    return url, 443 if ssl else 80, ssl


def _load_jsonl(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            chunks.append(json.loads(line))
    return chunks


def _upsert_batch_with_retry(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert one batch; retries up to MAX_RETRIES times on transient failures."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"\n    WARNING: batch failed (attempt {attempt}/{MAX_RETRIES}): {exc}"
                f" — retrying in {delay:.0f}s ...",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2


def _make_client(settings: object) -> chromadb.ClientAPI:
    """
    Return a ChromaDB client based on CHROMA_MODE env var.

    CHROMA_MODE=local (default) — PersistentClient, no server required.
    CHROMA_MODE=http            — HttpClient, requires ChromaDB server at chroma_url.
    """
    mode = os.environ.get("CHROMA_MODE", "local").strip().lower()

    if mode == "http":
        host, port, ssl = _parse_chroma_host_port(settings.chroma_url)
        print(f"Connecting to ChromaDB server at {settings.chroma_url} ...")
        client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        client.heartbeat()
        print("  Connected.")
        return client

    # local (default)
    persist_path = str(ROOT / settings.chroma_persist_dir.lstrip("./"))
    print(f"Using local ChromaDB at {persist_path} ...")
    return chromadb.PersistentClient(path=persist_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ChromaDB wellness_knowledge collection.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before seeding.",
    )
    args = parser.parse_args()

    settings = get_settings()
    openai_api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not CLEAN_DIR.exists():
        print(f"ERROR: Clean directory not found: {CLEAN_DIR}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(CLEAN_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {CLEAN_DIR}", file=sys.stderr)
        sys.exit(1)

    # --- Connect to ChromaDB ---
    try:
        client = _make_client(settings)
    except Exception as exc:
        print(f"ERROR: Cannot connect to ChromaDB — {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Embedding function ---
    ef = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL,
    )

    # --- Collection setup ---
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass  # didn't exist — fine

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' ready (current count: {collection.count()}).\n")

    # --- Seed ---
    module_counts: dict[str, int] = {}
    module_skipped: dict[str, int] = {}
    total_upserted = 0
    total_skipped = 0
    total_failed_batches = 0

    for path in jsonl_files:
        chunks = _load_jsonl(path)
        module_name = path.stem

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        seen_ids: dict[str, int] = {}  # id → occurrence count for collision detection
        skipped = 0

        for chunk in chunks:
            # Guard: skip empty text
            text = chunk.get("text", "").strip()
            if not text:
                print(f"  WARNING: skipping {chunk.get('handout_id')!r} — empty text")
                skipped += 1
                continue

            base_id = _make_chunk_id(
                chunk.get("module", module_name),
                chunk.get("handout_id", module_name),
            )

            # Disambiguate collisions within this file (e.g. split chunks sharing a handout_id)
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                chunk_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                chunk_id = base_id

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(
                {
                    "handout_id": chunk.get("handout_id", ""),
                    "module": chunk.get("module", module_name),
                    "skill_category": chunk.get("skill_category", ""),
                    "content_type": chunk.get("content_type", ""),
                    "char_count": chunk.get("char_count", len(text)),
                }
            )

        # Batch upsert with per-batch retry
        failed_batches = 0
        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_metas = metadatas[start:end]
            batch_num = start // BATCH_SIZE + 1
            print(
                f"  [{module_name}] batch {batch_num} ({len(batch_ids)} chunks) ...",
                end=" ",
                flush=True,
            )
            try:
                _upsert_batch_with_retry(collection, batch_ids, batch_docs, batch_metas)
                print("done")
            except Exception as exc:
                print(f"FAILED — {exc}")
                failed_batches += 1

        upserted = len(ids) - failed_batches * BATCH_SIZE
        module_counts[module_name] = upserted
        module_skipped[module_name] = skipped
        total_upserted += upserted
        total_skipped += skipped
        total_failed_batches += failed_batches

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SEED SUMMARY")
    print(f"  {'Module':<35} Upserted  Skipped")
    print(f"  {'-' * 50}")
    for module in sorted(module_counts):
        print(f"  {module:<35} {module_counts[module]:<9} {module_skipped[module]}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL upserted':<35} {total_upserted}")
    print(f"  {'TOTAL skipped (empty text)':<35} {total_skipped}")
    print(f"  {'Total in collection':<35} {collection.count()}")
    if total_failed_batches:
        print(f"  WARNING: {total_failed_batches} batch(es) failed after {MAX_RETRIES} retries")
    print(f"{'=' * 60}")

    # --- Retrieval test ---
    print(f'\nRetrieval test: "{TEST_QUERY}"')
    print(f"  Top {TOP_K} results:\n")
    try:
        results = collection.query(
            query_texts=[TEST_QUERY],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = 1.0 - dist  # cosine distance -> similarity
            snippet = doc[:200].replace("\n", " ")
            print(f"  [{rank}] {meta['handout_id']} | {meta['skill_category']}")
            print(f"       similarity={similarity:.3f}  content_type={meta['content_type']}")
            print(f"       \"{snippet}{'...' if len(doc) > 200 else ''}\"")
            print()
    except Exception as exc:
        print(f"  Retrieval test failed: {exc}")


if __name__ == "__main__":
    main()

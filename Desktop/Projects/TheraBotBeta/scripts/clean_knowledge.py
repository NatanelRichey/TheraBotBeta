"""
clean_knowledge.py — Clean labeled JSONL chunks for RAG ingestion.

Reads:  data/knowledge/labeled/*.jsonl
Writes: data/knowledge/labeled/clean/<same filename>

Rules applied in order:
  1  Skip if char_count < 80
  2  Skip if handout_id contains "Worksheet" (case-insensitive)
  3  Skip if text contains any interactive/form string (case-insensitive)
  4  Skip TOC summary descriptions (< 400 chars, no bullets/numbered steps)
  5  Skip pleasant-events numbered list (Emotion Regulation Handout 16, items ≥ 100)
  6  Skip addiction enumeration lists (Distress Tolerance Handout 16a, pure bullet list)
  7  Merge consecutive thin chunks (same handout_id + skill_category, both < 300 chars)
  8  Split oversized chunks (> 1800 chars) at double newlines into ≤ 900-char parts
  9  Recalculate char_count after any merge or split
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path("data/knowledge/labeled")
OUTPUT_DIR = INPUT_DIR / "clean"

RULE3_STRINGS: list[str] = [
    "due date:",
    "week starting:",
    "circle a number",
    "rate your level",
    "describe your experience",
    "write on the back",
    "fill out this sheet",
    "describe two crisis",
    "practice each",
    "www.guilford.com",
    "check off any of the following",
    "describe your practice",
    "describe your efforts",
    "how effective was",
]

RULE4_PREFIXES: tuple[str, ...] = (
    "this handout",
    "this skill",
    "defines ",
    "describes ",
    "explains ",
    "introduces ",
    "focuses on",
)

THIN_THRESHOLD = 300       # Rule 7: merge chunks below this
MERGE_CAP = 900            # Rule 7: do not merge if combined would exceed this
SPLIT_THRESHOLD = 1800     # Rule 8: split chunks above this
SPLIT_TARGET = 900         # Rule 8: target part size after split
MIN_CHARS = 80             # Rule 1


# ---------------------------------------------------------------------------
# Skip counters dataclass
# ---------------------------------------------------------------------------

@dataclass
class FileStats:
    filename: str
    input_count: int = 0
    skipped_rule1: int = 0
    skipped_rule2: int = 0
    skipped_rule3: int = 0
    skipped_rule4: int = 0
    skipped_rule5: int = 0
    skipped_rule6: int = 0
    merged_groups: int = 0      # number of merge operations performed
    chunks_merged_away: int = 0 # chunks consumed by merging
    split_count: int = 0        # chunks that were split
    output_count: int = 0


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------

def _text(chunk: dict) -> str:
    return chunk.get("text", "")


def rule1_too_short(chunk: dict) -> bool:
    return chunk.get("char_count", len(_text(chunk))) < MIN_CHARS


def rule2_worksheet(chunk: dict) -> bool:
    return "worksheet" in chunk.get("handout_id", "").lower()


def rule3_form_strings(chunk: dict) -> bool:
    text_lower = _text(chunk).lower()
    return any(s in text_lower for s in RULE3_STRINGS)


def rule4_toc_description(chunk: dict) -> bool:
    text = _text(chunk)
    if len(text) >= 400:
        return False
    text_lower = text.lower().lstrip()
    if not any(text_lower.startswith(p) for p in RULE4_PREFIXES):
        return False
    # Must contain no bullet points or numbered steps to be a TOC summary
    has_bullet = bool(re.search(r"^\s*[-•*]\s", text, re.MULTILINE))
    has_numbered_step = bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))
    return not (has_bullet or has_numbered_step)


def rule5_pleasant_events_list(chunk: dict) -> bool:
    if chunk.get("handout_id", "") != "Emotion Regulation Handout 16":
        return False
    text = _text(chunk)
    # Contains numbered items ≥ 100
    return bool(re.search(r"\b(100|10[1-9]|1[1-9]\d|\d{3,})\.", text))


def rule6_addiction_pure_bullets(chunk: dict) -> bool:
    if chunk.get("handout_id", "") != "Distress Tolerance Handout 16a":
        return False
    text = _text(chunk).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # Check that every non-empty line is a bullet item
    all_bullets = all(
        re.match(r"^[-•*]\s", line) or re.match(r"^\d+[.)]\s", line)
        for line in lines
    )
    if not all_bullets:
        return False
    # Has no full stops mid-text (i.e. no explanatory sentences)
    # Strip trailing periods on list items — look for ". " or ".\n" mid-string
    text_no_trailing = re.sub(r"\.\s*$", "", text, flags=re.MULTILINE)
    has_sentence = bool(re.search(r"\.\s", text_no_trailing))
    return not has_sentence


# ---------------------------------------------------------------------------
# Filter pass
# ---------------------------------------------------------------------------

def apply_filters(chunks: list[dict], stats: FileStats) -> list[dict]:
    kept: list[dict] = []
    for chunk in chunks:
        # Ensure char_count is present
        if "char_count" not in chunk:
            chunk["char_count"] = len(_text(chunk))

        if rule1_too_short(chunk):
            stats.skipped_rule1 += 1
            continue
        if rule2_worksheet(chunk):
            stats.skipped_rule2 += 1
            continue
        if rule3_form_strings(chunk):
            stats.skipped_rule3 += 1
            continue
        if rule4_toc_description(chunk):
            stats.skipped_rule4 += 1
            continue
        if rule5_pleasant_events_list(chunk):
            stats.skipped_rule5 += 1
            continue
        if rule6_addiction_pure_bullets(chunk):
            stats.skipped_rule6 += 1
            continue
        kept.append(chunk)
    return kept


# ---------------------------------------------------------------------------
# Rule 7 — Merge consecutive thin chunks
# ---------------------------------------------------------------------------

def _merge_pass(chunks: list[dict], stats: FileStats) -> tuple[list[dict], bool]:
    """One pass over the list; returns (new_list, changed)."""
    result: list[dict] = []
    i = 0
    changed = False
    while i < len(chunks):
        curr = chunks[i]
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            same_handout = curr.get("handout_id") == nxt.get("handout_id")
            same_skill = curr.get("skill_category") == nxt.get("skill_category")
            both_thin = (
                curr.get("char_count", 0) < THIN_THRESHOLD
                and nxt.get("char_count", 0) < THIN_THRESHOLD
            )
            combined_len = curr.get("char_count", 0) + nxt.get("char_count", 0)
            if same_handout and same_skill and both_thin and combined_len <= MERGE_CAP:
                merged_text = _text(curr) + "\n" + _text(nxt)
                merged = {**curr, "text": merged_text, "char_count": len(merged_text)}
                result.append(merged)
                stats.merged_groups += 1
                stats.chunks_merged_away += 1  # one chunk is consumed
                i += 2
                changed = True
                continue
        result.append(curr)
        i += 1
    return result, changed


def apply_merges(chunks: list[dict], stats: FileStats) -> list[dict]:
    changed = True
    while changed:
        chunks, changed = _merge_pass(chunks, stats)
    return chunks


# ---------------------------------------------------------------------------
# Rule 8 — Split oversized chunks
# ---------------------------------------------------------------------------

def _split_chunk(chunk: dict, stats: FileStats) -> list[dict]:
    text = _text(chunk)
    if len(text) <= SPLIT_THRESHOLD:
        return [chunk]

    parts = text.split("\n\n")
    if len(parts) == 1:
        warnings.warn(
            f"No natural split point for chunk handout_id={chunk.get('handout_id')!r}, "
            f"char_count={len(text)} — leaving as-is."
        )
        return [chunk]

    # Greedily group parts into buckets ≤ SPLIT_TARGET chars
    buckets: list[list[str]] = []
    current_bucket: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part)
        sep_len = 2 if current_bucket else 0  # "\n\n" separator
        if current_bucket and current_len + sep_len + part_len > SPLIT_TARGET:
            buckets.append(current_bucket)
            current_bucket = [part]
            current_len = part_len
        else:
            current_bucket.append(part)
            current_len += sep_len + part_len

    if current_bucket:
        buckets.append(current_bucket)

    if len(buckets) <= 1:
        # Splitting produced only one bucket — not actually useful
        return [chunk]

    stats.split_count += 1
    result: list[dict] = []
    for idx, bucket in enumerate(buckets):
        new_text = "\n\n".join(bucket)
        new_chunk = {
            **chunk,
            "text": new_text,
            "char_count": len(new_text),
        }
        if idx > 0:
            # Distinguish sub-chunks by appending a split index to chunk_id if present
            if "chunk_id" in new_chunk:
                new_chunk["chunk_id"] = f"{new_chunk['chunk_id']}_split{idx}"
        result.append(new_chunk)
    return result


def apply_splits(chunks: list[dict], stats: FileStats) -> list[dict]:
    result: list[dict] = []
    for chunk in chunks:
        result.extend(_split_chunk(chunk, stats))
    return result


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_dir: Path) -> FileStats:
    stats = FileStats(filename=input_path.name)

    raw_chunks: list[dict] = []
    with input_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            # Skip markdown fences (```json, ```) that some editors prepend
            if not line or line.startswith("```"):
                continue
            try:
                raw_chunks.append(json.loads(line))
            except json.JSONDecodeError:
                # Repair over-escaped quotes: \\" → \" (double-escape artifact)
                repaired = line.replace('\\\\"', '\\"')
                try:
                    raw_chunks.append(json.loads(repaired))
                except json.JSONDecodeError as e:
                    warnings.warn(
                        f"{input_path.name} line {lineno}: skipping malformed JSON — {e}\n"
                        f"  Content: {line[:120]!r}"
                    )

    stats.input_count = len(raw_chunks)

    # Rules 1-6: filter
    chunks = apply_filters(raw_chunks, stats)

    # Rule 7: merge
    chunks = apply_merges(chunks, stats)

    # Rule 8: split
    chunks = apply_splits(chunks, stats)

    # Rule 9: char_count is already maintained throughout

    stats.output_count = len(chunks)

    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return stats


def print_file_summary(stats: FileStats) -> None:
    skipped_total = (
        stats.skipped_rule1
        + stats.skipped_rule2
        + stats.skipped_rule3
        + stats.skipped_rule4
        + stats.skipped_rule5
        + stats.skipped_rule6
    )
    print(f"\n{'-' * 60}")
    print(f"File: {stats.filename}")
    print(f"  Input chunks  : {stats.input_count}")
    print(f"  Skipped total : {skipped_total}")
    print(f"    Rule 1 (too short)        : {stats.skipped_rule1}")
    print(f"    Rule 2 (worksheet)        : {stats.skipped_rule2}")
    print(f"    Rule 3 (form strings)     : {stats.skipped_rule3}")
    print(f"    Rule 4 (TOC description)  : {stats.skipped_rule4}")
    print(f"    Rule 5 (pleasant events)  : {stats.skipped_rule5}")
    print(f"    Rule 6 (addiction bullets): {stats.skipped_rule6}")
    print(f"  Merged (ops / chunks gone) : {stats.merged_groups} / {stats.chunks_merged_away}")
    print(f"  Split (chunks expanded)    : {stats.split_count}")
    print(f"  Output chunks : {stats.output_count}")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: list[FileStats] = []
    for path in jsonl_files:
        stats = process_file(path, OUTPUT_DIR)
        print_file_summary(stats)
        all_stats.append(stats)

    # Grand totals
    total_input = sum(s.input_count for s in all_stats)
    total_skipped = sum(
        s.skipped_rule1 + s.skipped_rule2 + s.skipped_rule3
        + s.skipped_rule4 + s.skipped_rule5 + s.skipped_rule6
        for s in all_stats
    )
    total_merged_ops = sum(s.merged_groups for s in all_stats)
    total_merged_away = sum(s.chunks_merged_away for s in all_stats)
    total_splits = sum(s.split_count for s in all_stats)
    total_output = sum(s.output_count for s in all_stats)

    print(f"\n{'=' * 60}")
    print("GRAND TOTAL")
    print(f"  Files processed : {len(all_stats)}")
    print(f"  Input chunks    : {total_input}")
    print(f"  Skipped         : {total_skipped}")
    print(f"  Merged ops      : {total_merged_ops}  (chunks consumed: {total_merged_away})")
    print(f"  Split chunks    : {total_splits}")
    print(f"  Output chunks   : {total_output}")
    print(f"{'=' * 60}")
    print(f"\nCleaned files written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

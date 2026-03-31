"""CLI viewer for TurnTrace records stored in Redis.

Usage:
    python scripts/view_traces.py --session-id <uuid>
    python scripts/view_traces.py --session-id <uuid> --last 5
    python scripts/view_traces.py --session-id <uuid> --turn 3
    python scripts/view_traces.py --user-id <uuid> --last 30
    python scripts/view_traces.py --session-id <uuid> --json
    python scripts/view_traces.py --session-id <uuid> --turn 3 --show-prompt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running from project root or scripts/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


async def main() -> None:
    parser = argparse.ArgumentParser(description="View TurnTrace records")
    parser.add_argument("--session-id", help="Session UUID to inspect")
    parser.add_argument("--user-id", help="User UUID — lists traces across all sessions")
    parser.add_argument("--turn", type=int, help="Load a specific turn number")
    parser.add_argument("--last", type=int, default=20, help="Max number of traces (default 20)")
    parser.add_argument("--json", action="store_true", dest="raw_json", help="Output raw JSON")
    parser.add_argument("--show-prompt", action="store_true", help="Print full system prompt")
    args = parser.parse_args()

    if not args.session_id and not args.user_id:
        parser.print_help()
        sys.exit(1)

    from app.services.turn_trace_store import get_turn_trace_store

    store = get_turn_trace_store()

    if args.turn and args.session_id:
        traces = []
        t = await store.get(args.session_id, args.turn)
        if t is None:
            print(f"No trace found for session={args.session_id} turn={args.turn}")
            return
        traces = [t]
    elif args.session_id:
        traces = await store.list_for_session(args.session_id, last_n=args.last)
    else:
        traces = await store.list_for_user(args.user_id, last_n=args.last)

    if not traces:
        print("No traces found.")
        return

    if args.raw_json:
        print(json.dumps([json.loads(t.model_dump_json()) for t in traces], indent=2))
        return

    SEP = "─" * 72

    for trace in traces:
        cost_str = f"${trace.llm_cost_usd:.6f}" if trace.llm_cost_usd is not None else "—"
        tokens_str = (
            f"{trace.llm_input_tokens} in / {trace.llm_output_tokens} out"
            if trace.llm_input_tokens is not None else "—"
        )
        path_str = " → ".join(trace.routing_path) if trace.routing_path else "—"
        skills_str = ", ".join(trace.retrieved_skill_categories) if trace.retrieved_skill_categories else "none"
        rag_str = (
            f"True ({trace.retrieved_knowledge_count} chunks — {skills_str})"
            if trace.rag_required else "False"
        )
        safety_str = f"{trace.safety_label} ({trace.safety_confidence:.2f})" if trace.safety_label else "—"
        mem_str = (
            f"wm={trace.had_working_memory}  "
            f"episodic={trace.episodic_count}  "
            f"longterm={trace.longterm_count}  "
            f"special={trace.had_special_instructions}"
        )
        alpha_str = (
            f"{'PASS' if trace.alpha_passed else 'FAIL'}  flags={trace.alpha_flags or []}"
            if trace.alpha_passed is not None else "—"
        )
        response_preview = trace.response[:120].replace("\n", " ") if trace.response else "—"

        print(f"Turn {trace.turn_number}  {trace.timestamp[:19]}Z  "
              f"session={trace.session_id[:8]}…  profile={trace.profile}")
        print(f"  Path    : {path_str}")
        print(f"  Safety  : {safety_str}   RAG: {rag_str}")
        print(f"  Memory  : {mem_str}")
        if trace.llm_model:
            print(f"  LLM     : {trace.llm_model}   {tokens_str}   {cost_str}")
        print(f"  Alpha   : {alpha_str}")
        print(f"  Cache   : {'HIT' if trace.cache_hit else 'MISS'}")
        print(f"  Response: \"{response_preview}\"")

        if args.show_prompt and trace.system_prompt:
            STAGE_SEP = "\n\n---\n\n"
            stages = trace.system_prompt.split(STAGE_SEP)
            print(f"\n  System prompt ({len(stages)} stages):")
            for i, stage in enumerate(stages, 1):
                print(f"\n  ── Stage {i} ──")
                for line in stage.splitlines():
                    print(f"  {line}")

        print(SEP)


if __name__ == "__main__":
    asyncio.run(main())

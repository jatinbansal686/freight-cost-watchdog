"""Command-line entry point.

    python -m watchdog.cli run    [--no-cache] [--explain-mode llm|template]
    python -m watchdog.cli repro  [--runs 3]
"""
from __future__ import annotations

import argparse
import shutil
import sys

from watchdog.config import REPO_ROOT, load_config
from watchdog.report import run_pipeline, write_outputs


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config()
    result = run_pipeline(cfg, explain_mode=args.explain_mode, use_cache=not args.no_cache)
    write_outputs(result, cfg)

    tracker = result["tracker"]
    print(f"Wrote {cfg.paths.output} ({len(result['output_df'])} rows)")
    print(f"LLM calls: {tracker.total_calls} | prompt tokens: {tracker.total_prompt_tokens} | completion tokens: {tracker.total_completion_tokens}")


def cmd_repro(args: argparse.Namespace) -> None:
    cfg = load_config()
    repro_dir = REPO_ROOT / "outputs" / "repro"
    if repro_dir.exists():
        shutil.rmtree(repro_dir)
    repro_dir.mkdir(parents=True)

    explain_cache = REPO_ROOT / "outputs" / "explanation_cache.json"

    run_hashes = []
    for i in range(1, args.runs + 1):
        run_dir = repro_dir / f"run{i}"
        run_dir.mkdir(parents=True)
        if explain_cache.exists():
            explain_cache.unlink()  # force the LLM to genuinely re-run prose each time

        result = run_pipeline(cfg, explain_mode="llm", use_cache=True, outputs_dir=run_dir)
        out_path = run_dir / "output.csv"
        import csv as csv_mod

        result["output_df"].to_csv(out_path, index=False, quoting=csv_mod.QUOTE_MINIMAL)
        run_hashes.append((i, out_path))
        print(f"run{i}: {len(result['output_df'])} rows, {result['tracker'].total_calls} LLM calls")

    numeric_cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    import pandas as pd

    dfs = [pd.read_csv(p)[numeric_cols] for _, p in run_hashes]
    identical = all(dfs[0].equals(d) for d in dfs[1:])

    report_lines = [
        "# Reproducibility check",
        "",
        f"Ran the pipeline {args.runs}x with `--explain-mode llm`, clearing the prose cache (not the note-enrichment",
        "cache -- that is deterministic factual extraction, not stylistic) between runs so the LLM genuinely",
        "re-generates every `reason` string each time.",
        "",
        f"**`route`, `week_of`, `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`, `flagged`, `matched_note_id`"
        f" identical across all {args.runs} runs: {'YES' if identical else 'NO'}**",
        "",
        "(`reason` wording may vary between runs -- see outputs/repro/run*/output.csv.)",
    ]
    (REPO_ROOT / "REPRO.md").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))

    if not identical:
        sys.exit(1)


PUBLISHED_RATE_PER_MILLION_INPUT_USD = 0.03   # openai/gpt-oss-20b via OpenRouter, checked 2026-08-19:
PUBLISHED_RATE_PER_MILLION_OUTPUT_USD = 0.13  # https://openrouter.ai/openai/gpt-oss-20b -- see README


def cmd_cost_log(args: argparse.Namespace) -> None:
    """One full COLD run (both caches cleared) over the entire shipment file, reporting exactly
    the four things the assignment asks for: LLM calls, input tokens, output tokens, rough cost."""
    cfg = load_config()
    for cache_name in ("notes_index.json", "explanation_cache.json"):
        cache_path = REPO_ROOT / "outputs" / cache_name
        if cache_path.exists():
            cache_path.unlink()

    result = run_pipeline(cfg, explain_mode="llm", use_cache=False)
    write_outputs(result, cfg)
    tracker = result["tracker"]

    input_cost = tracker.total_prompt_tokens / 1_000_000 * PUBLISHED_RATE_PER_MILLION_INPUT_USD
    output_cost = tracker.total_completion_tokens / 1_000_000 * PUBLISHED_RATE_PER_MILLION_OUTPUT_USD

    by_purpose = tracker.by_purpose()
    lines = [
        "# Token / cost log -- one full cold run",
        "",
        f"Model: `{cfg.llm.model}` via NVIDIA NIM (`{cfg.llm.base_url}`), `temperature=0`.",
        "",
        f"- LLM calls: **{tracker.total_calls}** ({', '.join(f'{v} {k}' for k, v in by_purpose.items())})",
        f"- Total input (prompt) tokens: **{tracker.total_prompt_tokens}**",
        f"- Total output (completion) tokens: **{tracker.total_completion_tokens}**",
        f"- Actual cost on the NVIDIA NIM free tier: **$0.00**",
        f"- Rough equivalent at a published `{cfg.llm.model}` rate "
        f"(${PUBLISHED_RATE_PER_MILLION_INPUT_USD}/1M input, ${PUBLISHED_RATE_PER_MILLION_OUTPUT_USD}/1M output tokens): "
        f"**${input_cost + output_cost:.4f}** (${input_cost:.4f} input + ${output_cost:.4f} output)",
    ]
    (REPO_ROOT / "outputs" / "token_cost_log.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="watchdog")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline once and write output.csv")
    run_p.add_argument("--no-cache", action="store_true", help="Ignore cached note enrichment/explanations")
    run_p.add_argument("--explain-mode", choices=["llm", "template"], default="llm")
    run_p.set_defaults(func=cmd_run)

    repro_p = sub.add_parser("repro", help="Run the pipeline N times and diff the results")
    repro_p.add_argument("--runs", type=int, default=3)
    repro_p.set_defaults(func=cmd_repro)

    cost_p = sub.add_parser("cost-log", help="One full cold run; writes outputs/token_cost_log.md")
    cost_p.set_defaults(func=cmd_cost_log)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

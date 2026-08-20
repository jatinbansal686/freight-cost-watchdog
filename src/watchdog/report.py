"""Orchestrates the full pipeline and writes output.csv (the submission) plus diagnostics."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd

from watchdog.baselines import add_baselines
from watchdog.config import Config
from watchdog.detect import candidates_only, flag_candidates
from watchdog.explain import ReasonContext, build_reason, find_near_miss, load_cache, save_cache
from watchdog.io_utils import load_notes, load_shipments
from watchdog.llm import CostTracker, LLMClient
from watchdog.notes.enrich import EnrichedNote, NoteCache
from watchdog.notes.gate import run_gate
from watchdog.notes.index import NotesIndex
from watchdog.weekly import aggregate_weekly

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def _fmt_pct(x: float | None, suffix: str) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x * 100:+.1f}% {suffix}"


def run_pipeline(
    cfg: Config,
    explain_mode: str = "llm",
    use_cache: bool = True,
    outputs_dir: Path | None = None,
    notes_cache_path: Path | None = None,
):
    """`outputs_dir` controls where this run's diagnostics (weekly_metrics.csv, retrieval_trace,
    explanation cache) land. `notes_cache_path` is separate and defaults to the committed
    `outputs/notes_index.json` regardless of `outputs_dir` -- note enrichment is deterministic
    factual extraction (temperature=0, validated, cached by note id), so an isolated run (e.g. one
    repro iteration) should still reuse it rather than silently re-enriching all 10 notes."""
    outputs_dir = outputs_dir or (cfg.paths.output.parent / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    notes_cache_path = notes_cache_path or (cfg.paths.output.parent / "outputs" / "notes_index.json")
    notes_cache_path.parent.mkdir(parents=True, exist_ok=True)

    shipments = load_shipments(cfg.paths.shipments)
    notes_df = load_notes(cfg.paths.notes)

    weekly = aggregate_weekly(shipments)
    baselined = add_baselines(weekly, cfg.baselines)
    flagged = flag_candidates(baselined, cfg.detector, cfg.baselines.own_history_weeks)
    candidates = candidates_only(flagged)

    tracker = CostTracker()
    llm_client = LLMClient(cfg.llm, tracker)

    # Retrieval is built first and needs no LLM call (notes/index.py embeds raw note text only).
    notes_index = NotesIndex(notes_df, top_k=cfg.retrieval.top_k)
    note_cache = NoteCache(notes_cache_path, use_cache=use_cache)
    notes_lookup = notes_df.set_index("note_id")
    enriched_by_id: dict[str, EnrichedNote] = {}

    explanation_cache_path = outputs_dir / "explanation_cache.json"
    explanation_cache = load_cache(explanation_cache_path) if use_cache else {}

    output_rows = []
    retrieval_trace = []

    for _, row in candidates.iterrows():
        route = row["route"]
        route_type = row["route_type"]
        week_of: date = row["week_of"].date()

        shortlist = notes_index.query(route, route_type, week_of.isoformat())

        # Lazy enrichment: only notes retrieval actually surfaced get an LLM call, and only on a
        # cache miss (NoteCache.get_or_enrich). A note never retrieved never reaches this line.
        shortlisted_notes = []
        for item in shortlist:
            note_id = item["note_id"]
            if note_id not in enriched_by_id:
                note_row = notes_lookup.loc[note_id]
                enriched_by_id[note_id] = note_cache.get_or_enrich(
                    llm_client,
                    note_id,
                    note_row["date"].date().isoformat(),
                    note_row["applies_to"],
                    note_row["note"],
                    max_tokens=cfg.notes.enrichment_max_tokens,
                    max_retries=cfg.notes.enrichment_max_retries,
                )
            shortlisted_notes.append(enriched_by_id[note_id])

        gate_result = run_gate(route, week_of, shortlisted_notes, cfg.notes.open_ended_note_weeks)

        matched_note = enriched_by_id[gate_result.matched_note_id] if gate_result.matched_note_id else None
        verdict = "justified" if matched_note else "unexplained"

        near_miss_note, near_miss_check = None, None
        if verdict == "unexplained":
            found = find_near_miss(gate_result.checks, enriched_by_id, week_of, cfg.notes.near_miss_days)
            if found:
                near_miss_note, near_miss_check = found

        own_pct = None if pd.isna(row["vs_own_history"]) else float(row["vs_own_history"])
        peer_pct = None if pd.isna(row["vs_similar_routes"]) else float(row["vs_similar_routes"])

        ctx = ReasonContext(
            route=route,
            week_of=week_of.isoformat(),
            cost_per_tonne_km=float(row["cost_per_tonne_km"]),
            vs_own_history_pct=own_pct,
            vs_similar_routes_pct=peer_pct,
            verdict=verdict,
            matched_note=matched_note,
            near_miss_note=near_miss_note,
            near_miss_check=near_miss_check,
        )
        reason = build_reason(ctx, mode=explain_mode, client=llm_client, cache=explanation_cache, max_tokens=cfg.explain.max_tokens)
        if row["low_confidence"]:
            reason += f" (Note: only {int(row['baseline_weeks_used'])} prior week(s) of history available for this route, below the usual 8-week baseline.)"

        output_rows.append(
            {
                "route": route,
                "week_of": week_of.isoformat(),
                "cost_per_tonne_km": round(row["cost_per_tonne_km"], 2),
                "vs_own_history": _fmt_pct(own_pct, "vs this route's past average"),
                "vs_similar_routes": _fmt_pct(peer_pct, "vs similar-length routes this week"),
                "flagged": "No (justified)" if verdict == "justified" else "Yes",
                "matched_note_id": matched_note.note_id if matched_note else "",
                "reason": reason,
            }
        )

        retrieval_trace.append(
            {
                "route": route,
                "week_of": week_of.isoformat(),
                "retrieved": shortlist,
                "gate_checks": [c.__dict__ for c in gate_result.checks],
                "matched_note_id": gate_result.matched_note_id,
            }
        )

    note_cache.save()

    if explain_mode != "template":
        save_cache(explanation_cache_path, explanation_cache)

    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).sort_values(["route", "week_of"]).reset_index(drop=True)

    weekly_metrics = flagged.copy()
    weekly_metrics["week_of"] = weekly_metrics["week_of"].dt.strftime("%Y-%m-%d")

    return {
        "output_df": output_df,
        "weekly_metrics_df": weekly_metrics,
        "retrieval_trace": retrieval_trace,
        "tracker": tracker,
    }


def write_outputs(result: dict, cfg: Config, outputs_dir: Path | None = None) -> None:
    outputs_dir = outputs_dir or (cfg.paths.output.parent / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    result["output_df"].to_csv(cfg.paths.output, index=False, quoting=csv.QUOTE_MINIMAL)
    result["weekly_metrics_df"].to_csv(outputs_dir / "weekly_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    with open(outputs_dir / "retrieval_trace.jsonl", "w") as f:
        for entry in result["retrieval_trace"]:
            f.write(json.dumps(entry) + "\n")

"""The assignment's "show a check": a small hand-labelled set scored against output.csv, plus a
zero-tolerance hallucination audit that independently re-derives every "No (justified)" verdict
from the committed note cache -- it does not just trust what the pipeline claims.

    python -m eval.run_eval
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from watchdog.notes.enrich import EnrichedNote  # noqa: E402
from watchdog.notes.gate import check_note  # noqa: E402
from watchdog.config import load_config  # noqa: E402

ALL_NOTE_IDS = [f"N{i:03d}" for i in range(1, 11)]


def load_enriched_notes() -> dict[str, EnrichedNote]:
    raw = json.loads((REPO_ROOT / "outputs" / "notes_index.json").read_text())
    return {nid: EnrichedNote(**fields) for nid, fields in raw.items()}


def score_against_labels(output: pd.DataFrame, expected: pd.DataFrame) -> int:
    merged = output.merge(expected, on=["route", "week_of"], how="outer", suffixes=("", "_expected"), indicator=True)
    mismatches = 0

    missing_from_output = merged[merged["_merge"] == "right_only"]
    if not missing_from_output.empty:
        print(f"MISSING from output.csv: {len(missing_from_output)} expected candidate row(s) not produced:")
        print(missing_from_output[["route", "week_of"]].to_string(index=False))
        mismatches += len(missing_from_output)

    unexpected_in_output = merged[merged["_merge"] == "left_only"]
    if not unexpected_in_output.empty:
        print(f"UNEXPECTED in output.csv: {len(unexpected_in_output)} row(s) not in the labelled set:")
        print(unexpected_in_output[["route", "week_of"]].to_string(index=False))
        mismatches += len(unexpected_in_output)

    both = merged[merged["_merge"] == "both"]
    for _, row in both.iterrows():
        got_note = "" if pd.isna(row["matched_note_id"]) else row["matched_note_id"]
        expected_note = "" if pd.isna(row["matched_note_id_expected"]) else row["matched_note_id_expected"]
        if row["flagged"] != row["flagged_expected"] or got_note != expected_note:
            print(
                f"MISMATCH {row['route']} {row['week_of']}: got flagged={row['flagged']!r} "
                f"matched_note_id={got_note!r}, expected flagged={row['flagged_expected']!r} "
                f"matched_note_id={expected_note!r}"
            )
            mismatches += 1

    return mismatches


def hallucination_audit(output: pd.DataFrame, notes_by_id: dict[str, EnrichedNote], open_ended_weeks: int) -> int:
    """For every 'No (justified)' row, independently re-check the cited note against the four gate
    conditions from the committed cache. Any failure is a fabricated citation -- fail the run."""
    justified = output[output["flagged"] == "No (justified)"]
    failures = 0
    for _, row in justified.iterrows():
        note_id = row["matched_note_id"]
        note = notes_by_id.get(note_id)
        if note is None:
            print(f"FABRICATED: {row['route']} {row['week_of']} cites {note_id!r} which does not exist")
            failures += 1
            continue

        week_of = date.fromisoformat(row["week_of"])
        check = check_note(note, row["route"], week_of, open_ended_weeks)
        if not check.passed:
            print(
                f"FABRICATED: {row['route']} {row['week_of']} cites {note_id} but it fails the gate on "
                f"re-check (route_match={check.route_match}, window_overlap={check.window_overlap}, "
                f"cost_increase_and_applies={check.cost_increase_and_applies}, evidence_ok={check.evidence_ok})"
            )
            failures += 1
    print(f"Hallucination audit: {len(justified)} justified rows checked, {failures} failure(s).")
    return failures


def citation_counts(output: pd.DataFrame) -> dict[str, int]:
    counts = {nid: 0 for nid in ALL_NOTE_IDS}
    for note_id in output["matched_note_id"].dropna():
        if note_id:
            counts[note_id] = counts.get(note_id, 0) + 1
    return counts


def main() -> None:
    cfg = load_config()
    output = pd.read_csv(cfg.paths.output, dtype={"matched_note_id": "string"})
    output["week_of"] = output["week_of"].astype(str)

    expected_path = REPO_ROOT / "eval" / "expected_verdicts.csv"
    expected = pd.read_csv(expected_path, dtype={"matched_note_id": "string"})
    expected["week_of"] = expected["week_of"].astype(str)

    print(f"Rows: {len(output)} | justified: {(output['flagged'] == 'No (justified)').sum()} | "
          f"unexplained: {(output['flagged'] == 'Yes').sum()}")
    print()

    print("=== Labelled-set check ===")
    mismatches = score_against_labels(output, expected)
    print(f"{len(expected) - mismatches}/{len(expected)} labelled rows match." if mismatches else "All labelled rows match.")
    print()

    print("=== Citation counts ===")
    counts = citation_counts(output)
    for note_id, count in counts.items():
        print(f"  {note_id}: {count}")
    print()

    print("=== Hallucination audit ===")
    notes_by_id = load_enriched_notes()
    failures = hallucination_audit(output, notes_by_id, cfg.notes.open_ended_note_weeks)

    if mismatches or failures:
        print(f"\nFAILED: {mismatches} labelled-set mismatch(es), {failures} hallucination failure(s).")
        sys.exit(1)
    print("\nPASSED.")


if __name__ == "__main__":
    main()

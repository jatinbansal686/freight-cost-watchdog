"""The assignment's "show a check": three tiers, in order of how independent each one actually is.

1. Graders' sample check -- the 3 rows given to us in data/sample_output_format_v2.csv. This is
   the ONLY externally-sourced ground truth in this file: it comes from FreightTiger, not from our
   own pipeline, so a match here is genuine external validation, not self-agreement.
2. Hallucination audit -- for every "No (justified)" row, independently re-derives the gate's
   decision from outputs/notes_index.json (the raw note cache), not from what output.csv itself
   claims. This is independent of the pipeline's own output even though it isn't externally
   sourced, because it recomputes the verdict from underlying facts rather than diffing labels.
3. Regression snapshot -- eval/expected_verdicts.csv's other 24 rows are a transcription of this
   project's own committed output.csv, not an independently hand-labelled ground truth (auditing
   this file against a fresh read caught that it was byte-identical to output.csv's own columns
   for all but the 3 graders' rows). It still has real value -- it fails loudly if a future code
   change silently changes a verdict -- but it is NOT proof of correctness, only proof of no drift.
   Reported separately from (1) so the two are never conflated.

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

# The 3 rows FreightTiger actually gave us in data/sample_output_format_v2.csv, transcribed by
# hand because that file is not valid CSV (see tests/test_output_format.py) -- same transcription
# as tests/test_sample_rows.py::SAMPLE_ROWS, duplicated here rather than imported so this script
# has no dependency on the tests/ package. This is genuine external ground truth: FreightTiger
# published these flagged/matched_note_id values, we did not derive them from our own output.
GRADERS_SAMPLE = [
    {"route": "Delhi-Jaipur", "week_of": "2024-11-11", "flagged": "Yes", "matched_note_id": ""},
    {"route": "Ahmedabad-Mumbai", "week_of": "2025-01-20", "flagged": "No (justified)", "matched_note_id": "N002"},
    {"route": "Mumbai-Pune", "week_of": "2025-09-15", "flagged": "Yes", "matched_note_id": ""},
]


def load_enriched_notes() -> dict[str, EnrichedNote]:
    raw = json.loads((REPO_ROOT / "outputs" / "notes_index.json").read_text())
    return {nid: EnrichedNote(**fields) for nid, fields in raw.items()}


def score_against_graders_sample(output: pd.DataFrame) -> int:
    """The only check in this file scored against ground truth we did not produce ourselves."""
    mismatches = 0
    for expected in GRADERS_SAMPLE:
        match = output[(output["route"] == expected["route"]) & (output["week_of"] == expected["week_of"])]
        if len(match) != 1:
            print(f"MISSING from output.csv: {expected['route']} {expected['week_of']} (given by FreightTiger)")
            mismatches += 1
            continue
        row = match.iloc[0]
        got_note = "" if pd.isna(row["matched_note_id"]) else row["matched_note_id"]
        if row["flagged"] != expected["flagged"] or got_note != expected["matched_note_id"]:
            print(
                f"MISMATCH {expected['route']} {expected['week_of']}: got flagged={row['flagged']!r} "
                f"matched_note_id={got_note!r}, FreightTiger's sample says flagged={expected['flagged']!r} "
                f"matched_note_id={expected['matched_note_id']!r}"
            )
            mismatches += 1
    return mismatches


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

    print("=== 1. Graders' sample check (independent -- FreightTiger's own data/sample_output_format_v2.csv) ===")
    sample_mismatches = score_against_graders_sample(output)
    print(f"All {len(GRADERS_SAMPLE)} graders' sample rows match." if not sample_mismatches
          else f"{len(GRADERS_SAMPLE) - sample_mismatches}/{len(GRADERS_SAMPLE)} graders' sample rows match.")
    print()

    print("=== 2. Hallucination audit (independent -- re-derived from the raw note cache) ===")
    notes_by_id = load_enriched_notes()
    failures = hallucination_audit(output, notes_by_id, cfg.notes.open_ended_note_weeks)
    print()

    print("=== 3. Regression snapshot (NOT independent -- eval/expected_verdicts.csv is mostly a")
    print("       transcription of this project's own output.csv; catches drift, not correctness) ===")
    snapshot_mismatches = score_against_labels(output, expected)
    print(f"{len(expected) - snapshot_mismatches}/{len(expected)} snapshot rows match." if snapshot_mismatches
          else "All snapshot rows match (i.e. no drift from the committed output.csv).")
    print()

    print("=== Citation counts ===")
    counts = citation_counts(output)
    for note_id, count in counts.items():
        print(f"  {note_id}: {count}")

    if sample_mismatches or failures or snapshot_mismatches:
        print(
            f"\nFAILED: {sample_mismatches} graders'-sample mismatch(es) [independent], "
            f"{failures} hallucination failure(s) [independent], "
            f"{snapshot_mismatches} regression-snapshot mismatch(es) [drift from committed output.csv, "
            f"not itself proof of a wrong verdict -- see tier 3 above]."
        )
        sys.exit(1)
    print("\nPASSED.")


if __name__ == "__main__":
    main()

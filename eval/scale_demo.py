"""Scale validation for lazy/retrieval-triggered note enrichment (notes/enrich.py, report.py).

Runs the REAL pipeline, unmodified, against a synthetic dataset that is fully independent from the
graded assignment data (data/shipment_records.csv, data/context_notes.csv): different city names,
different note_id prefix (SCALE-N... vs N0..), generated and committed separately under
tests/fixtures/scale_demo/. It hits the real LLM (no mocking) so this is a genuine end-to-end
check, not just a unit test.

The fixture is 7 synthetic routes x 14 weeks of shipments (392 rows) engineered so exactly 5
route-weeks clear the real candidacy thresholds (config.yaml: detector.*, unchanged), each with one
matching "signal" note among 51 total notes -- the other 46 are noise on routes that don't exist in
this shipments file at all, so they can never legitimately justify anything regardless of what
retrieval returns. This is the setup that actually exercises "a note retrieval never surfaces never
costs an LLM call": with `top_k=5` on a 51-note corpus, only a handful of notes are ever candidates
for enrichment, not all 51 -- the opposite of the real assignment's 10-note corpus, where
`top_k=10` retrieves everything (see README section "Enrichment is lazy, not eager").

Costs real (free-tier) API calls. Run manually:

    python -m eval.scale_demo
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from watchdog.config import load_config  # noqa: E402
from watchdog.io_utils import load_notes  # noqa: E402
from watchdog.report import run_pipeline, write_outputs  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "scale_demo"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "scale_demo"

SIGNAL_NOTES = {
    "Nagpur-Indore": "SCALE-N010",
    "Lucknow-Kanpur": "SCALE-N025",
    "Patna-Ranchi": "SCALE-N033",
    "Surat-Vadodara": "SCALE-N041",
    "Coimbatore-Madurai": "SCALE-N048",
}

# A handful of notes on routes/topics maximally distant from every candidate -- if retrieval and
# lazy enrichment work correctly, an LLM call must never happen for any of these.
MUST_NEVER_BE_ENRICHED = [
    "SCALE-N001",  # Shimla-Manali
    "SCALE-N002",  # Jodhpur-Bikaner
    "SCALE-N009",  # Jamshedpur-Dhanbad
    "SCALE-N020",  # Sagar-Jabalpur
    "SCALE-N030",  # Karimnagar-Khammam
    "SCALE-N040",  # Erode-Tiruppur
    "SCALE-N050",  # Junagadh-Porbandar
]


def main() -> None:
    cfg = load_config()
    notes_df = load_notes(FIXTURES_DIR / "notes.csv")
    total_notes = len(notes_df)

    scale_paths = replace(
        cfg.paths,
        shipments=FIXTURES_DIR / "shipments.csv",
        notes=FIXTURES_DIR / "notes.csv",
        output=OUTPUTS_DIR / "output.csv",
    )
    # top_k=5 on a 51-note corpus (vs. the real assignment's top_k=10 on its 10-note corpus, left
    # untouched in config.yaml) is what actually exercises retrieval genuinely filtering the corpus.
    scale_retrieval = replace(cfg.retrieval, top_k=5)
    scale_cfg = replace(cfg, paths=scale_paths, retrieval=scale_retrieval)

    notes_cache_path = OUTPUTS_DIR / "notes_index.json"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {total_notes} notes ({len(SIGNAL_NOTES)} signal + {total_notes - len(SIGNAL_NOTES)} noise), top_k={scale_retrieval.top_k}")
    print("Running full pipeline with real LLM calls (use_cache=False, cold)...\n")

    result = run_pipeline(scale_cfg, explain_mode="llm", use_cache=False, outputs_dir=OUTPUTS_DIR, notes_cache_path=notes_cache_path)
    write_outputs(result, scale_cfg, outputs_dir=OUTPUTS_DIR)
    output_df = result["output_df"]
    tracker = result["tracker"]

    enriched = json.loads(notes_cache_path.read_text())
    enriched_ids = set(enriched.keys())

    print(f"Candidates: {len(output_df)}")
    print(f"Notes actually enriched: {len(enriched_ids)} / {total_notes} -> {sorted(enriched_ids)}")
    print(f"LLM calls: {tracker.total_calls} ({', '.join(f'{v} {k}' for k, v in tracker.by_purpose().items())})")
    print()

    failures = []

    if len(enriched_ids) >= total_notes:
        failures.append(f"enriched {len(enriched_ids)}/{total_notes} notes -- expected far fewer than the full corpus")

    never_enriched_violations = [nid for nid in MUST_NEVER_BE_ENRICHED if nid in enriched_ids]
    if never_enriched_violations:
        failures.append(f"these notes were never supposed to be retrieved/enriched but were: {never_enriched_violations}")
    else:
        print(f"Confirmed never enriched (never retrieved): {MUST_NEVER_BE_ENRICHED}")

    missing_signal = [nid for nid in SIGNAL_NOTES.values() if nid not in enriched_ids]
    if missing_signal:
        failures.append(f"signal notes never got retrieved/enriched at all: {missing_signal}")

    print()
    print("=== Business-logic correctness at scale (the gate, unmodified) ===")
    for route, expected_note in SIGNAL_NOTES.items():
        row = output_df[output_df["route"] == route]
        if len(row) != 1:
            failures.append(f"{route}: expected exactly 1 candidate row, got {len(row)}")
            continue
        row = row.iloc[0]
        ok = row["flagged"] == "No (justified)" and row["matched_note_id"] == expected_note
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {route}: flagged={row['flagged']!r} matched_note_id={row['matched_note_id']!r} (expected {expected_note})")
        if not ok:
            failures.append(f"{route}: expected No (justified)/{expected_note}, got {row['flagged']!r}/{row['matched_note_id']!r}")

    unexpected_justified = output_df[
        (output_df["flagged"] == "No (justified)") & (~output_df["route"].isin(SIGNAL_NOTES))
    ]
    if not unexpected_justified.empty:
        failures.append(f"unexpected justified row(s) outside the 5 signal routes:\n{unexpected_justified[['route', 'week_of', 'matched_note_id']]}")

    print()
    if failures:
        print(f"FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(
        f"PASSED: only {len(enriched_ids)}/{total_notes} notes were ever sent to the LLM, all 5 signal "
        f"notes were found via retrieval and correctly justified their candidate, and the 7 maximally-"
        f"irrelevant notes checked never triggered a single LLM call."
    )


if __name__ == "__main__":
    main()

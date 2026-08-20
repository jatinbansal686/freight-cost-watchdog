"""Derivation check for the candidacy thresholds (config.yaml: detector.*).

README.md and WALKTHROUGH.md both assert the 7%/20% thresholds sit "roughly at the 97th
percentile of observed weekly changes across all 728 route-weeks" -- previously only asserted in
prose, with nothing in the repo a reader could run to check it. This script recomputes those
percentiles directly from the committed `outputs/weekly_metrics.csv` diagnostics (all 728
route-weeks, not just the 27 that became candidates) and prints where the configured thresholds
actually land, so the claim is independently verifiable rather than taken on faith.

This is a read-only, offline analysis -- no LLM calls, no network. Run manually:

    python -m eval.threshold_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from watchdog.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    metrics_path = REPO_ROOT / "outputs" / "weekly_metrics.csv"
    df = pd.read_csv(metrics_path)

    own_pct = cfg.detector.own_history_threshold_pct
    peer_pct = cfg.detector.peer_threshold_pct

    own_changes = df["vs_own_history"].dropna() * 100
    peer_changes = df["vs_similar_routes"].dropna() * 100

    own_rank = (own_changes < own_pct).mean() * 100
    peer_rank = (peer_changes < peer_pct).mean() * 100

    print(f"weekly_metrics.csv rows: {len(df)}")
    print(f"vs_own_history observations (non-null): {len(own_changes)}")
    print(f"vs_similar_routes observations (non-null): {len(peer_changes)}")
    print()
    print(f"configured own_history_threshold_pct = {own_pct}%")
    print(f"  -> sits at the {own_rank:.1f}th percentile of observed vs_own_history changes")
    print()
    print(f"configured peer_threshold_pct = {peer_pct}%")
    print(f"  -> sits at the {peer_rank:.1f}th percentile of observed vs_similar_routes changes")
    print()
    claimed = "~97th percentile"
    for name, rank in (("own_history", own_rank), ("peer", peer_rank)):
        note = "matches" if 95 <= rank <= 99 else "does NOT match"
        print(f"{name} threshold {note} the README/WALKTHROUGH claim of {claimed} (actual: {rank:.1f}th)")


if __name__ == "__main__":
    main()

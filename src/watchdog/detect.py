"""Suspicious-candidate detection.

IMPLEMENTATION CHOICE, NOT A FREIGHTTIGER REQUIREMENT: the assignment says "flag any route where
the cost is rising and looks out of the ordinary" but does not define a threshold. The numbers
below are ours (config.yaml `detector`), documented in README.md as our choice, not the
assignment's. Roughly the 97th percentile of observed weekly changes across all 728 route-weeks.

A row is a candidate when EITHER comparison clears its threshold (thresholds are positive, so
clearing either implies the cost is rising on that axis):

    candidate  <=>  vs_own_history >= own_history_threshold_pct/100
                     OR vs_similar_routes >= peer_threshold_pct/100

Rows with fewer than the full baseline window (baseline_weeks_used < own_history_weeks) are kept,
not suppressed -- the assignment says compute and disclose, not hide. `low_confidence` marks them
so `explain.py` can add a short caveat to the `reason`.
"""
from __future__ import annotations

import pandas as pd

from watchdog.config import DetectorConfig


def flag_candidates(baselined: pd.DataFrame, cfg: DetectorConfig, own_history_weeks: int) -> pd.DataFrame:
    df = baselined.copy()

    own_hit = df["vs_own_history"] >= (cfg.own_history_threshold_pct / 100)
    peer_hit = df["vs_similar_routes"] >= (cfg.peer_threshold_pct / 100)
    df["is_candidate"] = own_hit.fillna(False) | peer_hit.fillna(False)

    df["low_confidence"] = df["baseline_weeks_used"] < own_history_weeks

    return df


def candidates_only(flagged: pd.DataFrame) -> pd.DataFrame:
    return flagged[flagged["is_candidate"]].sort_values(["route", "week_of"]).reset_index(drop=True)

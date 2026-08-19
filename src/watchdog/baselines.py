"""Comparison baselines, computed exactly per the assignment's definitions -- not ours to improvise.

vs_own_history: trailing rolling average of the route's OWN weekly cost_per_tonne_km, using only
    weeks strictly before the current week (no look-ahead). If fewer than `own_history_weeks`
    prior weeks exist, use all that are available -- never pad or extrapolate. A route's first
    week has zero prior weeks, so vs_own_history is blank (NaN) there.

vs_similar_routes: same-week, unweighted average of cost_per_tonne_km across the OTHER routes
    sharing the same route_type. The route itself is excluded from its own peer average.
"""
from __future__ import annotations

import pandas as pd

from watchdog.config import BaselineConfig


def add_own_history_baseline(weekly: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    df = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    by_route = df.groupby("route")["cost_per_tonne_km"]
    df["own_history_baseline"] = by_route.transform(
        lambda s: s.rolling(window=cfg.own_history_weeks, min_periods=1).mean().shift(1)
    )
    weeks_used = by_route.transform(lambda s: s.expanding().count().shift(1)).fillna(0)
    df["baseline_weeks_used"] = weeks_used.clip(upper=cfg.own_history_weeks).astype(int)

    df["vs_own_history"] = (df["cost_per_tonne_km"] / df["own_history_baseline"]) - 1
    return df


def add_peer_baseline(weekly: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    df = weekly.copy()
    if cfg.peer_scope != "route_type":
        raise NotImplementedError(f"Unsupported peer_scope: {cfg.peer_scope}")

    group_cols = ["route_type", "week_of"]
    group_sum = df.groupby(group_cols)["cost_per_tonne_km"].transform("sum")
    group_count = df.groupby(group_cols)["cost_per_tonne_km"].transform("count")

    if cfg.peer_excludes_self:
        peer_sum = group_sum - df["cost_per_tonne_km"]
        peer_count = group_count - 1
    else:
        peer_sum = group_sum
        peer_count = group_count

    df["peer_count"] = peer_count
    df["peer_baseline"] = peer_sum / peer_count.replace(0, pd.NA)
    df["vs_similar_routes"] = (df["cost_per_tonne_km"] / df["peer_baseline"]) - 1
    return df


def add_baselines(weekly: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    df = add_own_history_baseline(weekly, cfg)
    df = add_peer_baseline(df, cfg)
    return df.sort_values(["route", "week_of"]).reset_index(drop=True)

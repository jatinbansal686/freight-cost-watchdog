import pandas as pd
import pytest

from watchdog.baselines import add_baselines
from watchdog.config import BaselineConfig


def make_weekly_row(route, route_type, week_of, cost_per_tonne_km):
    return {
        "route": route,
        "route_type": route_type,
        "week_of": pd.Timestamp(week_of),
        "freight_cost_inr": cost_per_tonne_km,  # values are arbitrary here; only cost_per_tonne_km matters
        "tonne_km": 1.0,
        "shipment_count": 1,
        "cost_per_tonne_km": cost_per_tonne_km,
    }


@pytest.fixture
def baseline_cfg():
    return BaselineConfig(own_history_weeks=8, peer_scope="route_type", peer_excludes_self=True)


def test_own_history_no_look_ahead(baseline_cfg):
    """A spike in a later week must not affect an earlier week's baseline."""
    weeks = ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
    costs = [10.0, 10.0, 10.0, 1000.0]  # spike only in the last week
    rows = [make_weekly_row("A-B", "Short", w, c) for w, c in zip(weeks, costs)]
    df = pd.DataFrame(rows)

    out = add_baselines(df, baseline_cfg)
    row0 = out[out["week_of"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert pd.isna(row0["vs_own_history"])  # first week: no prior weeks at all
    assert row0["baseline_weeks_used"] == 0

    row1 = out[out["week_of"] == pd.Timestamp("2024-01-08")].iloc[0]
    assert row1["own_history_baseline"] == 10.0
    assert row1["baseline_weeks_used"] == 1

    spike_row = out[out["week_of"] == pd.Timestamp("2024-01-22")].iloc[0]
    assert spike_row["own_history_baseline"] == 10.0  # unaffected by its own spike
    assert spike_row["baseline_weeks_used"] == 3


def test_own_history_fewer_than_8_prior_weeks_uses_all_available(baseline_cfg):
    weeks = [f"2024-0{m}-01" for m in range(1, 6)]  # 5 weeks -> only 4 prior weeks max
    rows = [make_weekly_row("A-B", "Short", w, 10.0 + i) for i, w in enumerate(weeks)]
    df = pd.DataFrame(rows)

    out = add_baselines(df, baseline_cfg)
    last_row = out.iloc[-1]
    assert last_row["baseline_weeks_used"] == 4  # not padded to 8, not extrapolated


def test_own_history_caps_at_configured_window(baseline_cfg):
    weeks = [f"2024-{m:02d}-01" for m in range(1, 11)]  # 10 weeks -> 9 prior weeks available for the last
    rows = [make_weekly_row("A-B", "Short", w, 10.0) for w in weeks]
    df = pd.DataFrame(rows)

    out = add_baselines(df, baseline_cfg)
    last_row = out.iloc[-1]
    assert last_row["baseline_weeks_used"] == baseline_cfg.own_history_weeks  # capped at 8, not 9


def test_peer_average_excludes_self(baseline_cfg):
    week = "2024-01-01"
    rows = [
        make_weekly_row("A-B", "Short", week, 10.0),
        make_weekly_row("C-D", "Short", week, 20.0),
        make_weekly_row("E-F", "Short", week, 30.0),
    ]
    df = pd.DataFrame(rows)

    out = add_baselines(df, baseline_cfg)
    row_ab = out[out["route"] == "A-B"].iloc[0]
    assert row_ab["peer_count"] == 2
    assert row_ab["peer_baseline"] == pytest.approx((20.0 + 30.0) / 2)
    assert row_ab["vs_similar_routes"] == pytest.approx((10.0 / 25.0) - 1)


def test_peer_average_only_within_same_route_type(baseline_cfg):
    week = "2024-01-01"
    rows = [
        make_weekly_row("A-B", "Short", week, 10.0),
        make_weekly_row("C-D", "Long", week, 999.0),  # different route_type -- must not count as a peer
    ]
    df = pd.DataFrame(rows)

    out = add_baselines(df, baseline_cfg)
    row_ab = out[out["route"] == "A-B"].iloc[0]
    assert row_ab["peer_count"] == 0
    assert pd.isna(row_ab["peer_baseline"])

import pandas as pd
import pytest

from watchdog.config import DetectorConfig
from watchdog.detect import flag_candidates


@pytest.fixture
def detector_cfg():
    return DetectorConfig(own_history_threshold_pct=7.0, peer_threshold_pct=20.0)


def make_row(vs_own, vs_peer, baseline_weeks_used=8):
    return {
        "route": "A-B",
        "week_of": pd.Timestamp("2024-01-01"),
        "cost_per_tonne_km": 1.0,
        "vs_own_history": vs_own,
        "vs_similar_routes": vs_peer,
        "baseline_weeks_used": baseline_weeks_used,
    }


def test_own_history_threshold_boundary(detector_cfg):
    df = pd.DataFrame(
        [
            make_row(0.07, -1.0),  # exactly at threshold -> candidate
            make_row(0.0699, -1.0),  # just under -> not a candidate
        ]
    )
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert out.iloc[0]["is_candidate"]
    assert not out.iloc[1]["is_candidate"]


def test_peer_threshold_boundary(detector_cfg):
    df = pd.DataFrame(
        [
            make_row(-1.0, 0.20),  # exactly at threshold -> candidate
            make_row(-1.0, 0.1999),  # just under -> not a candidate
        ]
    )
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert out.iloc[0]["is_candidate"]
    assert not out.iloc[1]["is_candidate"]


def test_either_condition_is_sufficient(detector_cfg):
    df = pd.DataFrame([make_row(0.08, -0.5), make_row(-0.5, 0.25)])
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert out["is_candidate"].all()


def test_falling_cost_is_never_a_candidate(detector_cfg):
    df = pd.DataFrame([make_row(-0.30, -0.30)])
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert not out.iloc[0]["is_candidate"]


def test_missing_own_history_falls_back_to_peer_only(detector_cfg):
    """A route's first week has no vs_own_history (NaN); candidacy must still work off vs_similar_routes."""
    df = pd.DataFrame([make_row(float("nan"), 0.25)])
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert out.iloc[0]["is_candidate"]


def test_low_confidence_flag_marks_short_baselines(detector_cfg):
    df = pd.DataFrame([make_row(0.08, -0.5, baseline_weeks_used=3)])
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert out.iloc[0]["low_confidence"]


def test_full_baseline_is_not_low_confidence(detector_cfg):
    df = pd.DataFrame([make_row(0.08, -0.5, baseline_weeks_used=8)])
    out = flag_candidates(df, detector_cfg, own_history_weeks=8)
    assert not out.iloc[0]["low_confidence"]


def test_real_data_yields_27_candidates(flagged_df):
    assert flagged_df["is_candidate"].sum() == 27

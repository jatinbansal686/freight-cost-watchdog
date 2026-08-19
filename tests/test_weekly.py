import pandas as pd

from watchdog.weekly import aggregate_weekly
from tests.conftest import make_shipment_row


def test_cost_per_tonne_km_is_pooled_not_mean_of_ratios():
    """The formula is sum(cost) / sum(tonnes*km), NOT the mean of each shipment's own cost/tonne-km.
    These give different answers whenever tonne-km varies between shipments in the same week."""
    rows = [
        make_shipment_row("S1", "A", "B", "Short", "2024-01-01", qty=10, dist=100, cost=1000),  # ratio 1.0
        make_shipment_row("S2", "A", "B", "Short", "2024-01-02", qty=20, dist=100, cost=3000),  # ratio 1.5
    ]
    df = pd.DataFrame(rows)
    weekly = aggregate_weekly(df)
    assert len(weekly) == 1

    pooled = (1000 + 3000) / (10 * 100 + 20 * 100)
    mean_of_ratios = (1.0 + 1.5) / 2
    assert pooled != mean_of_ratios

    assert weekly.iloc[0]["cost_per_tonne_km"] == pooled


def test_monday_sunday_bucketing():
    rows = [
        make_shipment_row("S1", "A", "B", "Short", "2024-01-03", qty=1, dist=1, cost=1),  # Wed
        make_shipment_row("S2", "A", "B", "Short", "2024-01-07", qty=1, dist=1, cost=1),  # Sun, same week
        make_shipment_row("S3", "A", "B", "Short", "2024-01-08", qty=1, dist=1, cost=1),  # Mon, next week
    ]
    df = pd.DataFrame(rows)
    weekly = aggregate_weekly(df)
    weeks = sorted(weekly["week_of"].dt.strftime("%Y-%m-%d").tolist())
    assert weeks == ["2024-01-01", "2024-01-08"]

    week1 = weekly[weekly["week_of"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert week1["shipment_count"] == 2


def test_year_boundary_week_2024_12_30():
    """2024-12-30 is a Monday; shipments on Dec 31 and Jan 1 belong to that same Mon-Sun week,
    despite crossing the calendar year boundary."""
    rows = [
        make_shipment_row("S1", "A", "B", "Short", "2024-12-30", qty=1, dist=1, cost=1),  # Mon
        make_shipment_row("S2", "A", "B", "Short", "2024-12-31", qty=1, dist=1, cost=1),  # Tue
        make_shipment_row("S3", "A", "B", "Short", "2025-01-01", qty=1, dist=1, cost=1),  # Wed
        make_shipment_row("S4", "A", "B", "Short", "2025-01-05", qty=1, dist=1, cost=1),  # Sun, same week
        make_shipment_row("S5", "A", "B", "Short", "2025-01-06", qty=1, dist=1, cost=1),  # Mon, next week
    ]
    df = pd.DataFrame(rows)
    weekly = aggregate_weekly(df)
    week = weekly[weekly["week_of"] == pd.Timestamp("2024-12-30")].iloc[0]
    assert week["shipment_count"] == 4

    next_week = weekly[weekly["week_of"] == pd.Timestamp("2025-01-06")].iloc[0]
    assert next_week["shipment_count"] == 1


def test_real_data_reproduces_sample_cost_values(shipments_df):
    weekly = aggregate_weekly(shipments_df)
    checks = {
        ("Delhi-Jaipur", "2024-11-11"): 4.17,
        ("Ahmedabad-Mumbai", "2025-01-20"): 3.29,
        ("Mumbai-Pune", "2025-09-15"): 3.98,
    }
    for (route, week), expected in checks.items():
        row = weekly[(weekly["route"] == route) & (weekly["week_of"] == pd.Timestamp(week))].iloc[0]
        assert round(row["cost_per_tonne_km"], 2) == expected

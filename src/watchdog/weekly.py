"""Weekly aggregation: buckets shipments into Mon-Sun weeks and computes pooled cost per tonne-km.

cost_per_tonne_km is POOLED (sum of costs / sum of tonne-km), not the mean of each shipment's own
ratio. Verified against all three sample_output_format_v2.csv rows -- the mean-of-ratios formula
gives 4.16/3.28/3.97 against the required 4.17/3.29/3.98.
"""
from __future__ import annotations

import pandas as pd


def add_week_of(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_of"] = (df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")).dt.normalize()
    return df


def aggregate_weekly(shipments: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (route, route_type, week_of) with pooled cost_per_tonne_km."""
    df = add_week_of(shipments)
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    grouped = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            freight_cost_inr=("freight_cost_inr", "sum"),
            tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
    )
    grouped["cost_per_tonne_km"] = grouped["freight_cost_inr"] / grouped["tonne_km"]
    return grouped.sort_values(["route", "week_of"]).reset_index(drop=True)

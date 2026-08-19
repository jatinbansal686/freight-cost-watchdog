"""Loads and validates the two input CSVs. Fails loudly on anything that would silently mispriced a listing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHIPMENT_DTYPES = {
    "shipment_id": "string",
    "origin": "string",
    "destination": "string",
    "route_type": "string",
    "material": "string",
    "quantity_tonnes": "float64",
    "distance_km": "float64",
    "freight_cost_inr": "float64",
    "transporter": "string",
}

REQUIRED_SHIPMENT_COLUMNS = list(SHIPMENT_DTYPES) + ["shipment_date"]


def load_shipments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=SHIPMENT_DTYPES, parse_dates=["shipment_date"])

    missing_cols = set(REQUIRED_SHIPMENT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"shipment_records.csv missing columns: {sorted(missing_cols)}")

    null_counts = df[REQUIRED_SHIPMENT_COLUMNS].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Null values found in required columns:\n{bad}")

    if (df["quantity_tonnes"] <= 0).any():
        raise ValueError("quantity_tonnes must be > 0 for every shipment")
    if (df["distance_km"] <= 0).any():
        raise ValueError("distance_km must be > 0 for every shipment")
    if (df["freight_cost_inr"] <= 0).any():
        raise ValueError("freight_cost_inr must be > 0 for every shipment")

    if df["shipment_id"].duplicated().any():
        dupes = df.loc[df["shipment_id"].duplicated(), "shipment_id"].tolist()
        raise ValueError(f"Duplicate shipment_id values: {dupes}")

    df["route"] = df["origin"] + "-" + df["destination"]

    route_type_map = df.groupby("route")["route_type"].nunique()
    inconsistent = route_type_map[route_type_map > 1]
    if not inconsistent.empty:
        raise ValueError(f"route_type is not consistent per route for: {inconsistent.index.tolist()}")

    return df


NOTE_DTYPES = {"note_id": "string", "applies_to": "string", "note": "string"}


def load_notes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=NOTE_DTYPES, parse_dates=["date"])
    required = {"note_id", "date", "applies_to", "note"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"context_notes.csv missing columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("context_notes.csv has null values in a required column")
    if df["note_id"].duplicated().any():
        raise ValueError(f"Duplicate note_id values: {df.loc[df['note_id'].duplicated(), 'note_id'].tolist()}")
    return df

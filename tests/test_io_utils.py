"""Tests for io_utils.py: input validation for shipment_records.csv and context_notes.csv.
Previously untested despite enforcing the brief's grouping guarantee directly -- "route_type field
already provided... you don't need to derive length buckets yourself" only holds if a route's
route_type is actually consistent across its shipments, which is exactly what
load_shipments' route_type check exists to catch.
"""
from __future__ import annotations

import pandas as pd
import pytest

from watchdog.io_utils import load_notes, load_shipments

SHIPMENT_COLUMNS = [
    "shipment_id", "origin", "destination", "route_type", "material",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date", "transporter",
]


def _write_shipments(tmp_path, rows):
    df = pd.DataFrame(rows, columns=SHIPMENT_COLUMNS)
    path = tmp_path / "shipments.csv"
    df.to_csv(path, index=False)
    return path


def _valid_row(shipment_id="S1", origin="Delhi", destination="Jaipur", route_type="Short", date="2025-01-06"):
    return [shipment_id, origin, destination, route_type, "Cement", 10.0, 100.0, 5000.0, date, "T1"]


def test_load_shipments_happy_path_builds_route_column(tmp_path):
    path = _write_shipments(tmp_path, [_valid_row()])
    df = load_shipments(path)
    assert df.loc[0, "route"] == "Delhi-Jaipur"
    assert pd.api.types.is_datetime64_any_dtype(df["shipment_date"])


def test_load_shipments_missing_required_column_raises(tmp_path):
    df = pd.DataFrame([_valid_row()], columns=SHIPMENT_COLUMNS).drop(columns=["distance_km"])
    path = tmp_path / "shipments.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_shipments(path)


def test_load_shipments_null_in_required_column_raises(tmp_path):
    row = _valid_row()
    row[1] = None  # origin
    path = _write_shipments(tmp_path, [row])
    with pytest.raises(ValueError, match="Null values"):
        load_shipments(path)


@pytest.mark.parametrize("col_index,label", [(5, "quantity_tonnes"), (6, "distance_km"), (7, "freight_cost_inr")])
def test_load_shipments_non_positive_numeric_fields_raise(tmp_path, col_index, label):
    row = _valid_row()
    row[col_index] = 0
    path = _write_shipments(tmp_path, [row])
    with pytest.raises(ValueError, match=label):
        load_shipments(path)


def test_load_shipments_negative_value_also_raises(tmp_path):
    row = _valid_row()
    row[6] = -50.0  # distance_km
    path = _write_shipments(tmp_path, [row])
    with pytest.raises(ValueError, match="distance_km"):
        load_shipments(path)


def test_load_shipments_duplicate_shipment_id_raises(tmp_path):
    rows = [_valid_row(shipment_id="DUPE"), _valid_row(shipment_id="DUPE", date="2025-01-07")]
    path = _write_shipments(tmp_path, rows)
    with pytest.raises(ValueError, match="Duplicate shipment_id"):
        load_shipments(path)


def test_load_shipments_inconsistent_route_type_for_same_route_raises(tmp_path):
    """The core guardrail this module exists for: route_type must be taken as given, and given
    consistently. A route quietly recorded as both 'Short' and 'Medium' across shipments must
    fail loudly, not have the pipeline silently pick one."""
    rows = [
        _valid_row(shipment_id="S1", route_type="Short"),
        _valid_row(shipment_id="S2", route_type="Medium", date="2025-01-07"),
    ]
    path = _write_shipments(tmp_path, rows)
    with pytest.raises(ValueError, match="route_type is not consistent"):
        load_shipments(path)


def test_load_shipments_same_route_type_across_shipments_is_fine(tmp_path):
    rows = [
        _valid_row(shipment_id="S1", route_type="Short"),
        _valid_row(shipment_id="S2", route_type="Short", date="2025-01-07"),
    ]
    path = _write_shipments(tmp_path, rows)
    df = load_shipments(path)
    assert len(df) == 2
    assert (df["route"] == "Delhi-Jaipur").all()


def test_load_shipments_different_routes_can_have_different_route_types(tmp_path):
    rows = [
        _valid_row(shipment_id="S1", origin="Delhi", destination="Jaipur", route_type="Short"),
        _valid_row(shipment_id="S2", origin="Mumbai", destination="Delhi", route_type="Long", date="2025-01-07"),
    ]
    path = _write_shipments(tmp_path, rows)
    df = load_shipments(path)  # must not raise
    assert set(df["route"]) == {"Delhi-Jaipur", "Mumbai-Delhi"}


# ---------------------------------------------------------------------------
# load_notes
# ---------------------------------------------------------------------------

NOTE_COLUMNS = ["note_id", "date", "applies_to", "note"]


def _write_notes(tmp_path, rows):
    df = pd.DataFrame(rows, columns=NOTE_COLUMNS)
    path = tmp_path / "notes.csv"
    df.to_csv(path, index=False)
    return path


def test_load_notes_happy_path(tmp_path):
    path = _write_notes(tmp_path, [["N1", "2025-01-01", "All Routes", "Diesel prices rose."]])
    df = load_notes(path)
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_notes_missing_column_raises(tmp_path):
    df = pd.DataFrame([["N1", "2025-01-01", "All Routes"]], columns=["note_id", "date", "applies_to"])
    path = tmp_path / "notes.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_notes(path)


def test_load_notes_null_value_raises(tmp_path):
    path = _write_notes(tmp_path, [["N1", "2025-01-01", None, "Diesel prices rose."]])
    with pytest.raises(ValueError, match="null values"):
        load_notes(path)


def test_load_notes_duplicate_note_id_raises(tmp_path):
    rows = [
        ["N1", "2025-01-01", "All Routes", "Diesel prices rose."],
        ["N1", "2025-02-01", "All Routes", "A different note reusing the same id."],
    ]
    path = _write_notes(tmp_path, rows)
    with pytest.raises(ValueError, match="Duplicate note_id"):
        load_notes(path)

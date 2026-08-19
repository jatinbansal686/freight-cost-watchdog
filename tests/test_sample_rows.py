"""sample_output_format_v2.csv cannot be parsed by pandas as-is (see test_output_format.py), so the
three example rows are hand-transcribed here from the raw file and checked by (route, week_of)."""
import pandas as pd
import pytest

SAMPLE_ROWS = [
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "cost_per_tonne_km": 4.17,
        "vs_own_history": "+35.5% vs this route's past average",
        "vs_similar_routes": "+21.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
    },
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "cost_per_tonne_km": 3.29,
        "vs_own_history": "+29.5% vs this route's past average",
        "vs_similar_routes": "+22.5% vs similar-length routes this week",
        "flagged": "No (justified)",
        "matched_note_id": "N002",
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "cost_per_tonne_km": 3.98,
        "vs_own_history": "+9.2% vs this route's past average",
        "vs_similar_routes": "+23.6% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
    },
]


@pytest.mark.parametrize("expected", SAMPLE_ROWS, ids=[r["route"] for r in SAMPLE_ROWS])
def test_sample_row_reproduced_exactly(output_df, expected):
    match = output_df[(output_df["route"] == expected["route"]) & (output_df["week_of"] == expected["week_of"])]
    assert len(match) == 1, f"expected exactly one row for {expected['route']} {expected['week_of']}"
    row = match.iloc[0]

    assert row["cost_per_tonne_km"] == expected["cost_per_tonne_km"]
    assert row["vs_own_history"] == expected["vs_own_history"]
    assert row["vs_similar_routes"] == expected["vs_similar_routes"]
    assert row["flagged"] == expected["flagged"]
    matched = "" if pd.isna(row["matched_note_id"]) else row["matched_note_id"]
    assert matched == expected["matched_note_id"]


def test_mumbai_pune_stays_unexplained_despite_open_ended_n003(output_df):
    """The key trap in the dataset: N003 (diesel, no explicit end date) must NOT be treated as
    active indefinitely, or this row would be wrongly justified by it."""
    row = output_df[(output_df["route"] == "Mumbai-Pune") & (output_df["week_of"] == "2025-09-15")].iloc[0]
    assert row["flagged"] == "Yes"
    assert "N003" not in row["reason"]

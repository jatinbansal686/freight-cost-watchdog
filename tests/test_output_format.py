import csv

import pandas as pd

from watchdog.report import OUTPUT_COLUMNS


def test_header_matches_sample(cfg):
    """sample_output_format_v2.csv is not valid CSV (row 4 has an unquoted comma -- see README), so
    we read just the header line directly rather than asking pandas to parse the whole file."""
    with open(cfg.paths.sample_format) as f:
        header = f.readline().strip().split(",")
    assert header == OUTPUT_COLUMNS


def test_output_columns_in_order(output_df):
    assert list(output_df.columns) == OUTPUT_COLUMNS


def test_output_round_trips_through_pandas(cfg, output_df, tmp_path):
    path = tmp_path / "output.csv"
    output_df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(output_df)
    assert list(reloaded.columns) == OUTPUT_COLUMNS


def test_blank_matched_note_id_when_unexplained(output_df):
    unexplained = output_df[output_df["flagged"] == "Yes"]
    assert not unexplained.empty
    assert unexplained["matched_note_id"].isna().all() or (unexplained["matched_note_id"] == "").all()


def test_justified_rows_have_a_matched_note_id(output_df):
    justified = output_df[output_df["flagged"] == "No (justified)"]
    assert not justified.empty
    for note_id in justified["matched_note_id"]:
        assert isinstance(note_id, str) and note_id.startswith("N")


def test_flagged_values_are_only_the_two_allowed_strings(output_df):
    assert set(output_df["flagged"].unique()) <= {"Yes", "No (justified)"}


def test_week_of_is_iso_date_string(output_df):
    for value in output_df["week_of"]:
        pd.Timestamp(value)  # raises if not parseable
        assert len(value) == 10 and value[4] == "-" and value[7] == "-"


def test_output_sorted_by_route_then_week(output_df):
    sorted_df = output_df.sort_values(["route", "week_of"]).reset_index(drop=True)
    assert output_df.reset_index(drop=True).equals(sorted_df)

"""One case per guardrail the assignment names. Every note here is a synthetic EnrichedNote so
these tests never call the LLM and never depend on how it happened to phrase anything."""
from datetime import date

import pytest

from watchdog.notes.enrich import EnrichedNote
from watchdog.notes.gate import run_gate


def note(note_id, applies_to, effective_from, effective_to, indicates_cost_increase, applies_to_dataset, raw_text="text", evidence_span="text"):
    return EnrichedNote(
        note_id=note_id,
        date=effective_from,
        applies_to=applies_to,
        raw_text=raw_text,
        effective_from=effective_from,
        effective_to=effective_to,
        indicates_cost_increase=indicates_cost_increase,
        applies_to_dataset=applies_to_dataset,
        evidence_span=evidence_span,
    )


def test_wrong_route_rejected():
    n = note("N100", "Delhi-Jaipur", "2024-01-01", "2024-01-31", True, True)
    result = run_gate("Mumbai-Pune", date(2024, 1, 8), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].route_match is False


def test_wrong_window_rejected():
    """N001-style: the note's window ends before the candidate week even though the route matches."""
    n = note("N001", "Chennai-Bangalore", "2025-02-24", "2025-03-08", True, True)
    result = run_gate("Chennai-Bangalore", date(2025, 3, 10), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].window_overlap is False


def test_costs_not_affected_rejected():
    """N005/N006/N008/N010-style: real event, right route/window, but explicitly no cost impact."""
    n = note("N005", "Mumbai-Delhi", "2024-07-29", "2024-08-05", False, True, raw_text="costs were not significantly affected")
    result = run_gate("Mumbai-Delhi", date(2024, 7, 29), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].cost_increase_and_applies is False


def test_improvement_or_normalisation_rejected():
    """N007/N009-style: conditions improved / returned to normal -- not a cost-rise reason either."""
    n = note("N009", "Chennai-Bangalore", "2025-03-17", "2025-03-24", False, True, raw_text="returned to normal conditions")
    result = run_gate("Chennai-Bangalore", date(2025, 3, 17), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].cost_increase_and_applies is False


def test_out_of_dataset_rejected():
    """N004-style: applies_to says 'All Routes' but the text says the affected routes aren't in
    this dataset -- applies_to_dataset must catch this even though the route field alone would pass."""
    n = note("N004", "All Routes", "2024-03-11", None, True, False, raw_text="not part of this dataset")
    result = run_gate("Mumbai-Pune", date(2024, 3, 11), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].route_match is True  # route/window would have passed
    assert result.checks[0].cost_increase_and_applies is False  # dataset check is what rejects it


def test_valid_match_accepted():
    """N002-style: right route, right window, real cost increase, verbatim evidence."""
    n = note(
        "N002",
        "Ahmedabad-Mumbai",
        "2025-01-20",
        "2025-01-20",
        True,
        True,
        raw_text="a temporary surcharge on the Ahmedabad-Mumbai corridor",
        evidence_span="a temporary surcharge",
    )
    result = run_gate("Ahmedabad-Mumbai", date(2025, 1, 20), [n], open_ended_weeks=8)
    assert result.matched_note_id == "N002"


def test_no_note_passes_yields_blank_match():
    n = note("N999", "Somewhere-Else", "2020-01-01", "2020-01-31", True, True)
    result = run_gate("Mumbai-Pune", date(2024, 1, 8), [n], open_ended_weeks=8)
    assert result.matched_note_id is None


def test_evidence_must_be_verbatim_substring():
    n = note("N100", "A-B", "2024-01-01", "2024-01-31", True, True, raw_text="fuel costs rose", evidence_span="fuel costs skyrocketed")
    result = run_gate("A-B", date(2024, 1, 8), [n], open_ended_weeks=8)
    assert result.matched_note_id is None
    assert result.checks[0].evidence_ok is False


def test_all_routes_note_matches_any_route():
    n = note("N003", "All Routes", "2025-05-05", None, True, True)
    result = run_gate("Mumbai-Pune", date(2025, 5, 12), [n], open_ended_weeks=8)
    assert result.matched_note_id == "N003"


def test_open_ended_window_expires_after_configured_weeks():
    """N003-style open-ended note: active within the window, expired well after it."""
    n = note("N003", "All Routes", "2025-05-05", None, True, True)

    still_active = run_gate("Mumbai-Delhi", date(2025, 6, 2), [n], open_ended_weeks=8)
    assert still_active.matched_note_id == "N003"

    expired = run_gate("Mumbai-Pune", date(2025, 9, 15), [n], open_ended_weeks=8)
    assert expired.matched_note_id is None


def test_route_specific_preferred_over_all_routes_on_tie():
    specific = note("N010", "A-B", "2024-01-01", "2024-01-31", True, True)
    general = note("N020", "All Routes", "2024-01-01", "2024-01-31", True, True)
    result = run_gate("A-B", date(2024, 1, 8), [general, specific], open_ended_weeks=8)
    assert result.matched_note_id == "N010"

"""The deterministic gate -- where trust is won. The LLM never sets `flagged` or `matched_note_id`.

A note may justify a candidate only if ALL FOUR conditions hold. Each maps directly to a failure
mode the assignment names:

  1. route match       -- note's `applies_to` equals the candidate's route, or is "All Routes".
                           Guards against: "Wrong route ... does not count."
  2. window overlap     -- note's [effective_from, effective_to] overlaps the candidate's Mon-Sun
                           week. An open-ended note (no explicit effective_to) gets a window of
                           `effective_from + open_ended_note_weeks` -- see README for why 8 weeks.
                           Guards against: "Wrong window ... does not count."
  3. real cost increase -- indicates_cost_increase AND applies_to_dataset are both true.
                           Guards against: "a note that says costs weren't affected does not count."
  4. verbatim evidence  -- evidence_span is an exact substring of the note's own text. Re-checked
                           here (not just at enrichment time) as defense in depth.
                           Guards against: nothing invented to fill a gap.

If more than one note passes (never happens in this dataset, but the tie-break must still be
deterministic so the verdict cannot depend on retrieval order): route-specific before "All
Routes", then the narrower window, then lexicographic note_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from watchdog.notes.enrich import EnrichedNote


@dataclass
class NoteCheck:
    note_id: str
    route_match: bool
    window_overlap: bool
    cost_increase_and_applies: bool
    evidence_ok: bool
    window_start: str
    window_end: str

    @property
    def passed(self) -> bool:
        return self.route_match and self.window_overlap and self.cost_increase_and_applies and self.evidence_ok


@dataclass
class GateResult:
    matched_note_id: str | None
    checks: list[NoteCheck]


def _resolve_window(note: EnrichedNote, open_ended_weeks: int) -> tuple[date, date]:
    start = date.fromisoformat(note.effective_from)
    if note.effective_to is not None:
        end = date.fromisoformat(note.effective_to)
    else:
        end = start + timedelta(weeks=open_ended_weeks)
    return start, end


def _week_span(week_of: date) -> tuple[date, date]:
    return week_of, week_of + timedelta(days=6)


def check_note(note: EnrichedNote, route: str, week_of: date, open_ended_weeks: int) -> NoteCheck:
    route_match = note.applies_to == route or note.applies_to == "All Routes"

    window_start, window_end = _resolve_window(note, open_ended_weeks)
    week_start, week_end = _week_span(week_of)
    window_overlap = window_start <= week_end and window_end >= week_start

    cost_increase_and_applies = note.indicates_cost_increase and note.applies_to_dataset
    evidence_ok = bool(note.evidence_span) and note.evidence_span in note.raw_text

    return NoteCheck(
        note_id=note.note_id,
        route_match=route_match,
        window_overlap=window_overlap,
        cost_increase_and_applies=cost_increase_and_applies,
        evidence_ok=evidence_ok,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )


def run_gate(
    candidate_route: str,
    candidate_week_of: date,
    shortlisted_notes: list[EnrichedNote],
    open_ended_weeks: int,
) -> GateResult:
    checks = [check_note(note, candidate_route, candidate_week_of, open_ended_weeks) for note in shortlisted_notes]

    passing = [c for c in checks if c.passed]
    if not passing:
        return GateResult(matched_note_id=None, checks=checks)

    def sort_key(c: NoteCheck) -> tuple:
        is_all_routes = 1 if next(n for n in shortlisted_notes if n.note_id == c.note_id).applies_to == "All Routes" else 0
        window_span_days = (date.fromisoformat(c.window_end) - date.fromisoformat(c.window_start)).days
        return (is_all_routes, window_span_days, c.note_id)

    passing.sort(key=sort_key)
    return GateResult(matched_note_id=passing[0].note_id, checks=checks)

"""Plain-English `reason` text. The verdict is already decided by notes/gate.py and passed in here
-- this module is never asked to decide anything, only to phrase it.

Two modes:
  llm       (default) the LLM phrases the gate's own facts into 1-2 sentences. Every candidate
            reason is validated before being accepted -- it must name the matched note's id and
            date (justified) or the near-miss note's id (near-miss) and no other note id, and
            every number in it must trace back to the row's own fields, the note's dates, or the
            note's text. A row that fails validation falls back to the template.
  template  zero LLM calls, purely deterministic. `--explain-mode template` produces identical
            verdicts and numbers to `llm` mode -- the reproducibility proof.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from watchdog.llm import LLMClient
from watchdog.notes.enrich import EnrichedNote
from watchdog.notes.gate import GateResult, NoteCheck

NOTE_ID_PATTERN = re.compile(r"\bN\d{3}\b")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class ReasonContext:
    route: str
    week_of: str
    cost_per_tonne_km: float
    vs_own_history_pct: float | None
    vs_similar_routes_pct: float | None
    verdict: str  # "justified" | "unexplained"
    matched_note: EnrichedNote | None
    near_miss_note: EnrichedNote | None
    near_miss_check: NoteCheck | None


def _pct(x: float | None) -> str | None:
    if x is None:
        return None
    return f"{x * 100:+.1f}"


def find_near_miss(
    checks: list[NoteCheck],
    notes_by_id: dict[str, EnrichedNote],
    week_of: date,
    near_miss_days: int,
) -> tuple[EnrichedNote, NoteCheck] | None:
    """Closest FAILING note by date within +/- near_miss_days, route-specific preferred."""
    failing = [c for c in checks if not c.passed]
    if not failing:
        return None

    def days_from_week(check: NoteCheck) -> int:
        note_date = date.fromisoformat(notes_by_id[check.note_id].date)
        return abs((note_date - week_of).days)

    within_window = [c for c in failing if days_from_week(c) <= near_miss_days]
    if not within_window:
        return None

    def sort_key(c: NoteCheck) -> tuple:
        route_specific = 0 if c.route_match else 1
        return (route_specific, days_from_week(c), c.note_id)

    within_window.sort(key=sort_key)
    best = within_window[0]
    return notes_by_id[best.note_id], best


def _near_miss_descriptor(note: EnrichedNote, check: NoteCheck) -> str:
    if not check.cost_increase_and_applies:
        if not note.applies_to_dataset:
            return "describes something not part of this dataset"
        return "reports no cost impact on this route"
    if not check.window_overlap:
        return f"covers {check.window_start} to {check.window_end}, which does not reach this week"
    if not check.route_match:
        return f"applies to {note.applies_to}, not this route"
    return "does not fully match"


def build_template_reason(ctx: ReasonContext) -> str:
    own = _pct(ctx.vs_own_history_pct)
    peer = _pct(ctx.vs_similar_routes_pct)
    trigger_bits = []
    if own is not None:
        trigger_bits.append(f"{own}% vs this route's own recent average")
    if peer is not None:
        trigger_bits.append(f"{peer}% vs similar routes this week")
    trigger = " and ".join(trigger_bits)

    if ctx.verdict == "justified":
        note = ctx.matched_note
        evidence = note.evidence_span.rstrip(".")
        return (
            f"Matches note {note.note_id} dated {note.effective_from}: {evidence}. "
            f"Cost is {trigger}, and this note explains it."
        )

    if ctx.near_miss_note is not None:
        note, check = ctx.near_miss_note, ctx.near_miss_check
        descriptor = _near_miss_descriptor(note, check)
        return (
            f"The closest note ({note.note_id}, dated {note.date}) {descriptor} -- it does not "
            f"explain a cost rise here. Cost is {trigger}. No genuine justification found; flagged for review."
        )

    return f"No matching note found for this route or date range. Cost is {trigger}. Flagged for review."


def _expected_note_ids(ctx: ReasonContext) -> set[str]:
    if ctx.verdict == "justified":
        return {ctx.matched_note.note_id}
    if ctx.near_miss_note is not None:
        return {ctx.near_miss_note.note_id}
    return set()


def _allowed_numbers(ctx: ReasonContext) -> set[str]:
    allowed: set[str] = set()
    for pct in (ctx.vs_own_history_pct, ctx.vs_similar_routes_pct):
        if pct is not None:
            allowed.add(f"{pct * 100:.1f}")
            allowed.add(f"{abs(pct * 100):.1f}")
            allowed.add(f"{pct * 100:.0f}")
    for note in (ctx.matched_note, ctx.near_miss_note):
        if note is None:
            continue
        allowed.update(NUMBER_PATTERN.findall(note.raw_text))
        allowed.update(NUMBER_PATTERN.findall(note.date))
        allowed.update(NUMBER_PATTERN.findall(note.effective_from))
        if note.effective_to:
            allowed.update(NUMBER_PATTERN.findall(note.effective_to))
        for part in note.date.split("-"):
            allowed.add(part.lstrip("0") or "0")
    for part in ctx.week_of.split("-"):
        allowed.add(part.lstrip("0") or "0")
        allowed.add(part)
    return allowed


def _validate_llm_reason(text: str, ctx: ReasonContext) -> bool:
    found_ids = set(NOTE_ID_PATTERN.findall(text))
    expected_ids = _expected_note_ids(ctx)
    if not found_ids.issubset(expected_ids):
        return False
    if ctx.verdict == "justified" and not expected_ids.issubset(found_ids):
        return False

    # Strip note-id tokens (e.g. "N002") before scanning for numbers -- otherwise the "002" inside
    # a note id it was explicitly told to mention gets flagged as an unrecognised invented number.
    text_without_note_ids = NOTE_ID_PATTERN.sub("", text)

    allowed_numbers = _allowed_numbers(ctx)
    for token in NUMBER_PATTERN.findall(text_without_note_ids):
        stripped = token.lstrip("0") or "0"
        if token in allowed_numbers or stripped in allowed_numbers:
            continue
        return False
    return True


LLM_SYSTEM_PROMPT = """You write a short, plain-English explanation (1-2 sentences) of a freight \
cost verdict that has ALREADY been decided. You do not decide anything -- only phrase the facts \
you are given. Never mention a note id other than the one given to you (if any). Never invent a \
number; use only the numbers given to you.

If you are given a matched_note_id or a near_miss_note_id, you MUST literally write that exact id \
(e.g. "N002") and its date somewhere in your sentence, in addition to describing what the note says."""


def _llm_prompt(ctx: ReasonContext) -> str:
    own = _pct(ctx.vs_own_history_pct)
    peer = _pct(ctx.vs_similar_routes_pct)
    lines = [
        f"route: {ctx.route}",
        f"week_of: {ctx.week_of}",
        f"vs_own_history: {own}%" if own else "vs_own_history: n/a (first week on record)",
        f"vs_similar_routes: {peer}%" if peer else "vs_similar_routes: n/a",
        f"verdict: {ctx.verdict}",
    ]
    if ctx.verdict == "justified":
        note = ctx.matched_note
        lines += [
            f"matched_note_id: {note.note_id}",
            f"matched_note_date: {note.effective_from}",
            f"matched_note_evidence: {note.evidence_span}",
        ]
    elif ctx.near_miss_note is not None:
        note, check = ctx.near_miss_note, ctx.near_miss_check
        lines += [
            f"near_miss_note_id: {note.note_id}",
            f"near_miss_note_date: {note.date}",
            f"near_miss_reason_it_failed: {_near_miss_descriptor(note, check)}",
        ]
    return "\n".join(lines)


def _cache_key(ctx: ReasonContext) -> str:
    payload = {
        "route": ctx.route,
        "week_of": ctx.week_of,
        "verdict": ctx.verdict,
        "own": ctx.vs_own_history_pct,
        "peer": ctx.vs_similar_routes_pct,
        "matched": ctx.matched_note.note_id if ctx.matched_note else None,
        "near_miss": ctx.near_miss_note.note_id if ctx.near_miss_note else None,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _clean(text: str) -> str:
    return re.sub(r"\.{2,}", ".", text)


def build_reason(
    ctx: ReasonContext,
    mode: str,
    client: LLMClient | None,
    cache: dict[str, str],
    max_tokens: int = 500,
) -> str:
    if mode == "template" or client is None or not client.available:
        return _clean(build_template_reason(ctx))

    key = _cache_key(ctx)
    if key in cache:
        return _clean(cache[key])

    llm_text = client.complete(LLM_SYSTEM_PROMPT, _llm_prompt(ctx), purpose="explanation", max_tokens=max_tokens)
    if llm_text and _validate_llm_reason(llm_text, ctx):
        cache[key] = llm_text
        return _clean(llm_text)

    return _clean(build_template_reason(ctx))


def load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))

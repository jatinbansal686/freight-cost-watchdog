"""LLM note understanding: one call per note, cached to outputs/notes_index.json.

Enrichment is lazy and retrieval-triggered (see `NoteCache.get_or_enrich`): a note is only ever
sent to the LLM once it has actually been returned by `notes/index.py`'s retrieval for some
candidate AND is not already in the cache. A note that retrieval never surfaces never causes an
LLM call. This keeps the architecture correct for a much larger corpus (cheap embedding retrieval
narrows a small relevant set before any LLM spend) even though, on this project's 10-note corpus
with `retrieval.top_k=10`, every note is retrieved for the first candidate anyway -- so a cold run
still enriches all 10, same as before, just triggered by retrieval instead of at startup.

The LLM extracts judgment calls that need reading the prose (does this describe a cost rise? does
it actually apply to routes in our dataset? what's the verbatim evidence?). It does NOT determine
`effective_from` -- that is always the note's own `date` column, set in code, because in this
10-note set every note that states an explicit start date states the same date already in `date`
(verified for all 10). This removes one whole class of possible hallucination without changing
any output. `applies_to` (the route) is likewise read straight from the CSV, never asked of the
LLM -- see notes/gate.py.

`applies_to_dataset` exists specifically to catch N004: its `applies_to` column literally says
"All Routes", but the prose says the affected routes "are not part of this dataset" -- only a
text read catches that contradiction.

Every field is validated before being trusted: effective_to must be a strict ISO date (or null),
`evidence_span` must be an exact verbatim substring of the note's own text. A violation raises --
we do not silently coerce or drop the field.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from watchdog.llm import LLMClient

SYSTEM_PROMPT = """You extract structured facts from a single logistics operations note. \
Output ONLY a JSON object, no markdown fences, no commentary, with exactly these keys:
{"effective_to": "YYYY-MM-DD or null", "indicates_cost_increase": true/false, \
"applies_to_dataset": true/false, "evidence_span": "verbatim substring of the note text"}

Rules:
- effective_to: the date the situation described ended, ONLY if the note's own text explicitly \
names that end/return-to-normal date. Do not treat the `date:` field above as the end date unless \
the text itself states that same date as when things ended. Otherwise null.
- indicates_cost_increase: true only if the note describes something that would raise freight \
cost (fuel, tolls, weather disruption, surcharge, capacity shortage). If the note says costs were \
NOT affected, or conditions improved/normalized/absorbed with no rate change, this is false.
- applies_to_dataset: false ONLY if the note EXPLICITLY states that the affected routes or \
situation are not part of this dataset, or describes something clearly unrelated to freight \
routes altogether (e.g. an unrelated industry event). Do NOT set this false just because the note \
says there was no cost impact, no rate change, or costs were absorbed -- that is what \
indicates_cost_increase is for, not this field. Default to true.
- evidence_span: a short VERBATIM substring copied exactly from the note text, character for \
character, that best supports your indicates_cost_increase judgment. Do not paraphrase."""


@dataclass
class EnrichedNote:
    note_id: str
    date: str
    applies_to: str
    raw_text: str
    effective_from: str
    effective_to: str | None
    indicates_cost_increase: bool
    applies_to_dataset: bool
    evidence_span: str


class NoteValidationError(ValueError):
    pass


def _validate_iso_date(value: str, field_name: str, note_id: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise NoteValidationError(f"{note_id}: {field_name}={value!r} is not a strict ISO date") from exc


def _validate_and_build(note_id: str, note_date: str, applies_to: str, raw_text: str, parsed: dict) -> EnrichedNote:
    required_keys = {"indicates_cost_increase", "applies_to_dataset", "evidence_span"}
    missing = required_keys - set(parsed)
    if missing:
        raise NoteValidationError(f"{note_id}: LLM response missing keys {missing}")

    effective_to = parsed.get("effective_to")
    if effective_to is not None:
        _validate_iso_date(effective_to, "effective_to", note_id)
        if effective_to < note_date:
            raise NoteValidationError(f"{note_id}: effective_to {effective_to} is before effective_from {note_date}")

    if not isinstance(parsed["indicates_cost_increase"], bool):
        raise NoteValidationError(f"{note_id}: indicates_cost_increase must be a boolean")
    if not isinstance(parsed["applies_to_dataset"], bool):
        raise NoteValidationError(f"{note_id}: applies_to_dataset must be a boolean")

    evidence_span = parsed["evidence_span"]
    if not evidence_span or evidence_span not in raw_text:
        raise NoteValidationError(f"{note_id}: evidence_span {evidence_span!r} is not a verbatim substring of the note")

    return EnrichedNote(
        note_id=note_id,
        date=note_date,
        applies_to=applies_to,
        raw_text=raw_text,
        effective_from=note_date,
        effective_to=effective_to,
        indicates_cost_increase=parsed["indicates_cost_increase"],
        applies_to_dataset=parsed["applies_to_dataset"],
        evidence_span=evidence_span,
    )


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def enrich_note(
    client: LLMClient, note_id: str, note_date: str, applies_to: str, raw_text: str, max_tokens: int = 600
) -> EnrichedNote:
    user_prompt = f"note_id: {note_id}\ndate: {note_date}\nnote text: {raw_text}"

    last_error: Exception | None = None
    for _ in range(2):
        content = client.complete(SYSTEM_PROMPT, user_prompt, purpose="note_enrichment", max_tokens=max_tokens)
        if content is None:
            raise NoteValidationError(f"{note_id}: LLM call failed (no content returned)")
        try:
            parsed = _parse_json_response(content)
            return _validate_and_build(note_id, note_date, applies_to, raw_text, parsed)
        except (json.JSONDecodeError, NoteValidationError) as exc:
            last_error = exc
    raise NoteValidationError(f"{note_id}: failed to produce a valid enrichment after retries: {last_error}")


class NoteCache:
    """Per-note-id enrichment cache -- the only thing that can trigger an LLM call.

    Same on-disk format as before (outputs/notes_index.json): a flat {note_id: {...fields...}}
    dict, one entry per note. `get_or_enrich` is called only for notes retrieval has actually
    surfaced for a candidate; a note never retrieved never reaches this method, so it never
    reaches `enrich_note`, so it never costs an LLM call.
    """

    def __init__(self, cache_path: Path, use_cache: bool = True):
        self.cache_path = cache_path
        self._cached: dict[str, dict] = {}
        if use_cache and cache_path.exists():
            self._cached = json.loads(cache_path.read_text())
        self._changed = False

    def get_or_enrich(
        self, client: LLMClient, note_id: str, note_date: str, applies_to: str, raw_text: str, max_tokens: int = 600
    ) -> EnrichedNote:
        if note_id in self._cached:
            return EnrichedNote(**self._cached[note_id])

        enriched = enrich_note(client, note_id, note_date, applies_to, raw_text, max_tokens=max_tokens)
        self._cached[note_id] = asdict(enriched)
        self._changed = True
        return enriched

    def save(self) -> None:
        if not self._changed:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cached, indent=2, sort_keys=True))

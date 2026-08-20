"""Proves enrichment is retrieval-triggered and lazy, not eager at startup (notes/enrich.py,
notes/index.py). Uses a tiny synthetic 3-note corpus with top_k=1, not the committed 10-note
dataset -- on the real dataset top_k=10 retrieves every note (see README), so laziness can only be
observed with a corpus larger than top_k. `enrich_note` is monkeypatched where we only need to
count/attribute calls; the one exception (test 4) calls the real `enrich_note` to prove validation
still fires on the lazy path.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from watchdog.notes.enrich import EnrichedNote, NoteCache, NoteValidationError
from watchdog.notes.index import NotesIndex

@pytest.fixture(autouse=True)
def _isolated_chroma_collection():
    """chromadb's EphemeralClient shares one process-wide "context_notes" collection by name
    (notes/index.py) across every NotesIndex instance in this pytest session -- including the real
    10-note one the session-scoped `pipeline_result` fixture builds elsewhere. These tests build a
    NotesIndex over a different, synthetic corpus, so the collection must be reset before and after
    each test here to avoid leaking into (or being contaminated by) that real collection."""
    import chromadb

    def _reset():
        try:
            chromadb.EphemeralClient().delete_collection("context_notes")
        except Exception:
            pass

    _reset()
    yield
    _reset()


NOTES_DF = pd.DataFrame(
    {
        "note_id": ["N1", "N2", "N3"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
        "applies_to": ["Mumbai-Delhi", "Chennai-Bangalore", "Ahmedabad-Mumbai"],
        "note": [
            "Diesel prices rose sharply affecting the Mumbai-Delhi corridor this week.",
            "Heavy flooding on the Chennai-Bangalore highway caused major delays.",
            "A temporary festival surcharge applies on the Ahmedabad-Mumbai route.",
        ],
    }
)


def _fake_enrich_note(calls: list[str]):
    def _enrich(client, note_id, note_date, applies_to, raw_text, max_tokens, max_retries):
        calls.append(note_id)
        return EnrichedNote(
            note_id=note_id,
            date=note_date,
            applies_to=applies_to,
            raw_text=raw_text,
            effective_from=note_date,
            effective_to=None,
            indicates_cost_increase=True,
            applies_to_dataset=True,
            evidence_span=raw_text[:10],
        )

    return _enrich


def test_startup_builds_retrieval_index_without_llm_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("watchdog.notes.enrich.enrich_note", _fake_enrich_note(calls))

    NotesIndex(NOTES_DF, top_k=1)  # indexing/embedding only -- no enrichment call

    assert calls == []


def test_retrieval_triggers_enrichment_only_for_the_retrieved_note(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr("watchdog.notes.enrich.enrich_note", _fake_enrich_note(calls))

    index = NotesIndex(NOTES_DF, top_k=1)
    cache = NoteCache(tmp_path / "notes_index.json", use_cache=True)

    shortlist = index.query("Mumbai-Delhi", "Long", "2025-01-06")
    lookup = NOTES_DF.set_index("note_id")
    for item in shortlist:
        row = lookup.loc[item["note_id"]]
        cache.get_or_enrich(
            object(), item["note_id"], row["date"].date().isoformat(), row["applies_to"], row["note"],
            max_tokens=600, max_retries=2,
        )

    assert calls == ["N1"]  # exactly one call, for the note retrieval actually surfaced
    assert "N2" not in calls  # never retrieved -> never enriched
    assert "N3" not in calls


def test_cached_note_is_not_re_enriched(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr("watchdog.notes.enrich.enrich_note", _fake_enrich_note(calls))
    cache_path = tmp_path / "notes_index.json"

    cache = NoteCache(cache_path, use_cache=True)
    cache.get_or_enrich(
        object(), "N003", "2025-05-05", "All Routes", "Diesel prices rose nationwide", max_tokens=600, max_retries=2
    )
    cache.get_or_enrich(
        object(), "N003", "2025-05-05", "All Routes", "Diesel prices rose nationwide", max_tokens=600, max_retries=2
    )
    cache.save()
    assert calls == ["N003"]  # second retrieval within the same run was a cache hit

    # A later run loading the persisted cache must also skip the LLM for the same note_id.
    calls.clear()
    reloaded = NoteCache(cache_path, use_cache=True)
    reloaded.get_or_enrich(
        object(), "N003", "2025-05-05", "All Routes", "Diesel prices rose nationwide", max_tokens=600, max_retries=2
    )
    assert calls == []


class _FakeClient:
    def __init__(self, content: str):
        self._content = content
        self.call_count = 0

    def complete(self, system, user, purpose, max_tokens=None):
        self.call_count += 1
        return self._content


def test_invalid_evidence_still_rejected_through_the_lazy_path(tmp_path):
    bad_response = json.dumps(
        {
            "effective_to": None,
            "indicates_cost_increase": True,
            "applies_to_dataset": True,
            "evidence_span": "text that does not appear in the note",
        }
    )
    cache = NoteCache(tmp_path / "notes_index.json", use_cache=True)

    with pytest.raises(NoteValidationError):
        cache.get_or_enrich(
            _FakeClient(bad_response), "N999", "2025-01-01", "All Routes", "unrelated note text",
            max_tokens=600, max_retries=2,
        )


def test_max_retries_is_respected_not_hardcoded(tmp_path):
    """Regression: enrich_note used to have a bare `for _ in range(2):` regardless of what the
    caller asked for. Passing max_retries=4 here must mean 4 real attempts, not 2."""
    bad_response = json.dumps(
        {
            "effective_to": None,
            "indicates_cost_increase": True,
            "applies_to_dataset": True,
            "evidence_span": "text that does not appear in the note",
        }
    )
    cache = NoteCache(tmp_path / "notes_index.json", use_cache=True)
    fake_client = _FakeClient(bad_response)

    with pytest.raises(NoteValidationError):
        cache.get_or_enrich(
            fake_client, "N999", "2025-01-01", "All Routes", "unrelated note text",
            max_tokens=600, max_retries=4,
        )
    assert fake_client.call_count == 4

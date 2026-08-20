"""Tests for notes/index.py. The default-top_k pin is a fast signature check (no embedding model
load); the behavioral tests reuse the same isolated-Chroma-collection pattern as
test_lazy_enrichment.py since NotesIndex always talks to a real (local, offline) Chroma instance.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from watchdog.notes.index import NotesIndex


def test_top_k_default_is_ten():
    """Regression pin: this was a stale 5 even though report.py always passed 10 explicitly (see
    README/config.yaml's "top_k=10 = every note in this corpus"). The class default must agree
    with what the pipeline actually uses and what the docs claim, not silently diverge from it."""
    sig = inspect.signature(NotesIndex.__init__)
    assert sig.parameters["top_k"].default == 10


@pytest.fixture(autouse=True)
def _isolated_chroma_collection():
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


def test_query_with_default_top_k_returns_every_note_in_a_small_corpus():
    """With only 3 notes and top_k defaulting to 10, every note should come back ranked -- top_k
    is a ceiling, not a fixed count."""
    index = NotesIndex(NOTES_DF)
    shortlist = index.query("Mumbai-Delhi", "Long", "2025-01-06")
    assert {item["note_id"] for item in shortlist} == {"N1", "N2", "N3"}


def test_query_respects_a_smaller_explicit_top_k():
    index = NotesIndex(NOTES_DF, top_k=1)
    shortlist = index.query("Mumbai-Delhi", "Long", "2025-01-06")
    assert len(shortlist) == 1


def test_query_ranks_the_matching_route_note_first():
    index = NotesIndex(NOTES_DF, top_k=3)
    shortlist = index.query("Mumbai-Delhi", "Long", "2025-01-06")
    assert shortlist[0]["note_id"] == "N1"
    assert shortlist[0]["distance"] is not None

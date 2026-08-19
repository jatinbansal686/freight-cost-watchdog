"""Vector retrieval over the 10 context notes -- the RAG layer.

Deliberately simple: with only 10 notes this demonstrates the intended retrieval architecture and
hands the deterministic gate a ranked shortlist. It is not tuned or benchmarked. ChromaDB with its
built-in local ONNX all-MiniLM-L6-v2 embeddings -- open source, runs locally, no API cost.

Retrieval proposes, the gate disposes: this module never decides a verdict, only ranks candidates
for the gate to check. If Chroma or its embedding model can't be loaded (e.g. no network on first
run to fetch the ~80MB ONNX model), we fall back to handing the gate all 10 notes unranked -- the
gate is the sole decision authority, so this cannot change any verdict, only the retrieval_trace.
"""
from __future__ import annotations

import pandas as pd


class NotesIndex:
    def __init__(self, notes_df: pd.DataFrame, top_k: int = 5):
        self.notes_df = notes_df
        self.top_k = top_k
        self._collection = None
        self._fallback = False
        self._build()

    def _build(self) -> None:
        try:
            import chromadb

            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(name="context_notes")
            collection.add(
                ids=self.notes_df["note_id"].tolist(),
                documents=self.notes_df["note"].tolist(),
                metadatas=[
                    {"note_id": r["note_id"], "date": r["date"].date().isoformat(), "applies_to": r["applies_to"]}
                    for _, r in self.notes_df.iterrows()
                ],
            )
            self._collection = collection
        except Exception as exc:  # noqa: BLE001 -- any embedding/model-download failure triggers fallback
            print(f"[notes/index] Chroma retrieval unavailable ({exc}); falling back to all-notes shortlist.")
            self._fallback = True

    def query(self, route: str, route_type: str, week_of: str) -> list[dict]:
        """Returns a ranked shortlist: [{note_id, distance}], best match first.

        distance is None in fallback mode (all notes, unranked)."""
        if self._fallback or self._collection is None:
            return [{"note_id": nid, "distance": None} for nid in self.notes_df["note_id"].tolist()]

        query_text = f"route {route}, route type {route_type}, week of {week_of}: reason for cost change"
        result = self._collection.query(query_texts=[query_text], n_results=min(self.top_k, len(self.notes_df)))
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [{"note_id": nid, "distance": dist} for nid, dist in zip(ids, distances)]

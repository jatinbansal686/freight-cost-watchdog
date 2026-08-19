# Reproducibility check

Ran the pipeline 3x with `--explain-mode llm`, clearing the prose cache (not the note-enrichment
cache -- that is deterministic factual extraction, not stylistic) between runs so the LLM genuinely
re-generates every `reason` string each time.

**`route`, `week_of`, `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`, `flagged`, `matched_note_id` identical across all 3 runs: YES**

(`reason` wording may vary between runs -- see outputs/repro/run*/output.csv.)

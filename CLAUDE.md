# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FreightTiger take-home case study: a pipeline that flags per-route freight-cost weeks that look
abnormally high, checks a small set of free-text "context notes" via RAG to see if a real
disruption (flood, festival surcharge, diesel spike, etc.) explains the rise, and only marks a
week `No (justified)` if a note survives a deterministic gate — never on the LLM's say-so. Full
design rationale lives in `README.md`; a live-demo script is in `WALKTHROUGH.md`. Read `README.md`
before changing thresholds, the gate, or the note-window logic — most "obvious" changes there are
deliberate judgment calls with a documented reason, not oversights.

`output.csv` at the repo root is the graded submission.

## Commands

```bash
pip install -r requirements.txt
export PYTHONPATH=src                                    # required outside pytest -- not pip-installed
cp .env.example .env                                  # fill in NVIDIA_API_KEY (only needed for LLM-mode runs)

python -m watchdog.cli run                             # full run -> output.csv (uses caches, calls LLM for anything not cached)
python -m watchdog.cli run --explain-mode template      # zero LLM calls, identical verdicts/numbers to LLM mode
python -m watchdog.cli run --no-cache                   # ignore note/explanation caches, force a cold run
python -m watchdog.cli repro --runs 3                    # N repro runs, prose cache cleared each time -> REPRO.md
python -m watchdog.cli cost-log                         # one full cold run (both caches cleared) -> outputs/token_cost_log.md
python -m watchdog.cli ask "question"                    # Q&A over output.csv/weekly_metrics.csv, one-shot
python -m watchdog.cli ask                                # same, interactive REPL
python -m eval.run_eval                                 # 3 checks: graders' 3-row sample (independent), hallucination audit (independent), regression snapshot (drift only, not correctness)

pytest                                                   # full suite, no API key needed (runs in template explain-mode)
pytest tests/test_gate.py                                # one file
pytest tests/test_gate.py::test_wrong_route_rejected -v  # one test
```

`pytest` config (`pythonpath = ["src"]`, `testpaths = ["tests"]`) is in `pyproject.toml`, not
`setup.cfg`. Only `run` with a cleared cache and `cost-log` ever hit the network/LLM; `ask` hits it
once per question (`purpose="qa"` in the cost tracker, not part of the graded pipeline's cost log);
everything else works offline against the committed `outputs/notes_index.json` note-enrichment
cache.

## Architecture

Linear pipeline, one module per stage, orchestrated by `report.run_pipeline`:

```
data/shipment_records.csv, data/context_notes.csv
        |
        v
weekly.aggregate_weekly       Mon-Sun weeks, POOLED cost_per_tonne_km (sum/sum, not mean-of-ratios)
        v
baselines.add_baselines       trailing 8-week own history (no look-ahead) + same-week route_type peer avg
        v
detect.flag_candidates        candidacy thresholds (config.yaml: detector.*) -> "suspicious" route-weeks
        v
notes.index.NotesIndex        ChromaDB (local ONNX embeddings) ranks all 10 notes per candidate
        v
notes.enrich.build_notes_index  one LLM call per note (cached by note_id in outputs/notes_index.json):
                                 does it describe a cost rise? apply to this dataset? verbatim evidence quote?
        v
notes.gate.run_gate           DETERMINISTIC, no LLM call — the only place `flagged`/`matched_note_id` are decided
        v
explain.build_reason          LLM (or --explain-mode template) phrases the reason from the gate's own facts only
        v
report.write_outputs          output.csv + outputs/weekly_metrics.csv + outputs/retrieval_trace.jsonl
```

**The invariant that must never be broken:** the LLM proposes (interprets note prose, ranks
retrieval, phrases sentences); `notes/gate.py` alone decides `flagged` and `matched_note_id`, and
has zero LLM calls in it. If a change makes the LLM's output flow into those two columns without
passing through `run_gate`'s four conditions, that's a regression regardless of how well it scores
manually — `eval/run_eval.py`'s hallucination audit re-derives every justified verdict independently
from the note cache specifically to catch this class of bug.

The four gate conditions (`notes/gate.py::check_note`), all required: route match (or "All
Routes"), date-window overlap with the candidate's Mon-Sun week, `indicates_cost_increase AND
applies_to_dataset` both true, and `evidence_span` a verbatim substring of the note's own text.

### Module map

- `weekly.py` — shipments -> weekly per-route aggregates. `cost_per_tonne_km` is pooled, verified
  digit-for-digit against `data/sample_output_format_v2.csv`.
- `baselines.py` — `vs_own_history` / `vs_similar_routes`, per the assignment's own definitions
  (`config.yaml: baselines.*` — not ours to change).
- `detect.py` — candidacy thresholds (`config.yaml: detector.*` — **our** judgment call, not
  assignment-specified; see README §4).
- `notes/index.py` — RAG retrieval. `top_k` is deliberately 10 (every note in this corpus); a
  smaller shortlist previously dropped a real match (README §6) — don't "optimize" this back down
  without re-reading why.
- `notes/enrich.py` — LLM note understanding, cached per `note_id`. Factual extraction at
  temperature=0, so this cache is reused across repro/eval runs unlike the prose cache — but
  temperature=0 only minimizes variance on the hosted API, it does not guarantee bit-identical
  output (see README §11 Limitations).
- `notes/gate.py` — the sole decision authority; see invariant above.
- `explain.py` — reason-string phrasing only, two modes (`llm` / `template`); has its own cache
  (`outputs/explanation_cache.json`) that `repro`/`cost-log` intentionally clear.
- `llm.py` — single wrapper around the NVIDIA NIM (OpenAI-compatible) client; all calls go through
  here so `CostTracker` can log call/token counts by purpose.
- `report.py` — orchestrates the above and writes `output.csv` + diagnostics.
- `ask.py` — Q&A stretch goal (`cli.py`'s `ask` command). Reads the artefacts `run` already wrote
  (`output.csv`, `outputs/weekly_metrics.csv`, `data/context_notes.csv`) rather than re-running the
  pipeline. Same "code decides scope, LLM only phrases" split as the rest of the project:
  `parse_question`/`build_context` deterministically resolve a route + optional month/year and
  filter to matching rows (no match -> fixed message, zero LLM calls); `answer_question` phrases
  from those rows only, then always appends a deterministic per-route "Ground truth" footer
  (`_verdict_line`, sourced straight from `output.csv`) beneath the LLM's prose -- added after live
  testing caught the model both dropping a route from a multi-route answer and misstating a
  justified/unexplained count, neither of which `notes/gate.py`'s fixed conditions can catch since
  free-form prose isn't as small a surface to verify. See README §7.
- `config.py` / `config.yaml` — one typed `Config` loaded from YAML + `.env`; distinguishes
  assignment-specified values (`baselines`) from this project's own judgment calls (`detector`,
  `notes`, `retrieval`, `qa`), each commented in `config.yaml` accordingly. No tunable numeric
  parameter should ever be a bare literal at its call site instead of a `config.yaml` field —
  `ask.py`'s LLM budget (`qa.max_tokens`) was originally hardcoded as `max_tokens=400` directly in
  the `client.complete()` call; moved into config for the same reason `detector.*`/`notes.*` are:
  a judgment call should be visible and changeable in one place, not buried in a function body.

### Data/output layout

- `data/` — the three input files (`shipment_records.csv`, `context_notes.csv`,
  `sample_output_format_v2.csv`) plus the case-study PDF. `sample_output_format_v2.csv` (not the
  `_v2`-less name mentioned in the brief's prose) is the binding format reference — it's the only
  one with `matched_note_id`, and it does not parse as valid CSV as-is (see README §4).
- `outputs/notes_index.json` — committed note-enrichment cache; lets everything except a cold
  `run`/`cost-log` work with **no API key**.
- `outputs/weekly_metrics.csv`, `outputs/retrieval_trace.jsonl` — diagnostics covering all 728
  route-weeks and full per-candidate retrieval rankings, not just the 27 that made it to
  `output.csv`.
- `outputs/repro/` — gitignored; regenerated by `cli repro`.

### Testing notes

- `tests/test_sample_rows.py` hand-transcribes the 3 sample rows instead of parsing
  `sample_output_format_v2.csv` with pandas, because that file itself isn't valid CSV.
- `tests/test_gate.py` has one test per gate guardrail — when adding a new gate condition, add a
  test here in the same pattern rather than folding it into an existing case.
- The full suite runs via the `pipeline_result` fixture in `tests/conftest.py`, which uses
  `explain_mode="template"` against the committed note cache — no network, no API key, and no LLM
  nondeterminism to work around.

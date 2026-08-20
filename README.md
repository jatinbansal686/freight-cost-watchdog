# Freight Cost Watchdog

A submission for FreightTiger's 24-hour AI Intern case study: *"build a smart assistant that spots
when a route's cost is rising in a way that doesn't look justified, and explains why in plain
English."*

**`output.csv` at the repo root is the graded submission.** This README maps 1:1 onto the brief's
own structure so it's fast to check off: what was asked, what we built, how, and why.

Result on the supplied data: **27 candidate route-weeks out of 728 total, 9 justified, 18
unexplained.** Only notes `N001`, `N002`, `N003` are ever cited as a match; the other seven are
cited zero times, which `eval/run_eval.py` verifies, not just asserts.

---

## 1. How to run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src                                 # required — package isn't pip-installed
cp .env.example .env                                   # fill in NVIDIA_API_KEY (only for LLM-mode)

python -m watchdog.cli run                              # full run -> output.csv
python -m watchdog.cli run --explain-mode template       # zero LLM calls, identical verdicts/numbers
python -m watchdog.cli run --no-cache                    # ignore caches, force a cold run
python -m watchdog.cli repro --runs 3                     # 3x run -> REPRO.md
python -m watchdog.cli cost-log                          # one full cold run -> outputs/token_cost_log.md
python -m watchdog.cli ask "why did X get pricier in <month>?"   # Q&A, see §5.4
python -m eval.run_eval                                    # 3 independent correctness checks, see §6
pytest                                                     # 139 tests, no API key needed
```

Everything except a cold `run`/`cost-log` works with **no API key** — note enrichment is cached at
`outputs/notes_index.json` (committed), and `--explain-mode template` produces byte-identical
verdicts and numbers from pure code, zero LLM calls.

---

## 2. Architecture — how data flows

```
shipment_records.csv, context_notes.csv
        |
        v
[1] weekly aggregation     Mon-Sun weeks, POOLED cost/tonne-km (sum/sum, not mean-of-ratios)
        v
[2] baselines               trailing 8-week own history (no look-ahead) + same-week peer avg
        v
[3] candidate detection     our own thresholds (config.yaml) -> "suspicious" route-weeks
        v
[4] RAG retrieval           ChromaDB, ranks all 10 notes per candidate      <- AI layer starts
        v
[5] LLM note understanding  lazy, one call per note, cached by note_id
        v
[6] DETERMINISTIC GATE      plain Python, zero LLM calls — the ONLY place a verdict is decided
        v
[7] LLM explanation          phrases the gate's own facts into a sentence
        v
    output.csv + diagnostics
```

**The one rule that must never break:** the LLM *proposes* (reads note prose, ranks retrieval,
phrases sentences); `notes/gate.py` alone *decides* `flagged` and `matched_note_id`, using plain
deterministic Python with no LLM call inside it. `eval/run_eval.py`'s hallucination audit
independently re-derives every justified verdict straight from the note cache — not from what
`output.csv` claims — specifically to catch a regression in this rule.

Module map: `weekly.py` (aggregation) → `baselines.py` (own-history / peer comparisons) →
`detect.py` (candidacy thresholds) → `notes/index.py` (retrieval) → `notes/enrich.py` (LLM note
understanding, cached) → `notes/gate.py` (the decision) → `explain.py` (LLM phrasing, cached
separately) → `report.py` (writes `output.csv`). `ask.py` is the standalone Q&A layer (§5.4);
`llm.py` is the one wrapper every LLM call goes through, so cost can be tracked centrally.

---

## 3. Core requirements — what the brief asked for, exactly

| Brief's requirement | What we built | Verified how |
|---|---|---|
| Group by route + given `route_type` | `weekly.py` groups on `(route, route_type)` as supplied — no re-derivation | — |
| `cost/tonne-km = total cost / (qty_tonnes × distance_km)` | Pooled: `sum(cost) / sum(qty × distance)` across all shipments in the route-week — **not** the mean of each shipment's own ratio | Matches all 3 rows of `data/sample_output_format_v2.csv` to the exact digit (`tests/test_sample_rows.py`) |
| Weeks are Mon–Sun, `week_of` = Monday | `shipment_date` bucketed to its Monday before any aggregation | `tests/test_weekly.py` (incl. year-boundary case) |
| vs. own history: trailing 8-week rolling avg, strictly prior weeks, no look-ahead, disclose if <8 weeks | Implemented exactly as specified. `outputs/weekly_metrics.csv`'s `baseline_weeks_used` column shows the exact count for every row | Two rows in `output.csv` run on <8 weeks (Delhi-Chennai 2024-01-08: 1 prior week; 2024-02-05: 5 prior weeks) — both explicitly flagged `low_confidence` in their `reason` text |
| vs. similar routes: same-week avg across other routes of the same `route_type`, self excluded | `baselines.py::vs_similar_routes`, self-exclusion via `peer_count` column | `Short`={Delhi-Jaipur, Mumbai-Pune}, `Long`={Delhi-Chennai, Mumbai-Delhi} — 1 peer each; `Medium` (3 routes) — 2 peers each |
| Flag rising, out-of-ordinary cost | `detect.py` candidacy thresholds — **our own judgment call**, not assignment-specified (see §7) | 27/728 route-weeks qualify |

---

## 4. Output format — the contract

Column names, order, and types match `data/sample_output_format_v2.csv` exactly:
`route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id, reason`.

Two format notes worth flagging honestly:

- **The supplied sample file is named `sample_output_format_v2.csv`, not `sample_output_format.csv`**
  as the brief's prose says. Only one such file was actually supplied, and it's the only one
  containing `matched_note_id` (which the brief calls part of the contract) — so it's treated as
  binding.
- **That file doesn't parse as valid CSV as supplied** — row 4's `reason` field has an unquoted
  comma against a header with zero quoting anywhere in the file, so `pandas.read_csv` raises a
  `ParserError` on it. Our own `output.csv` is written with `csv.QUOTE_MINIMAL`, which round-trips
  through pandas cleanly. (`tests/test_output_format.py` reads just the header line directly rather
  than parsing the whole sample file for this reason.)
- `vs_own_history` / `vs_similar_routes` are written as formatted strings
  (`"+29.5% vs this route's past average"`), matching the exact format FreightTiger's own sample
  uses for those columns — even though the brief calls them "numeric fields we grade." The raw
  floats aren't lost: they're in `outputs/weekly_metrics.csv` for anyone grading numerically.

---

## 5. The AI layer — RAG, LLM usage, and caching

This is a **RAG pipeline with deterministic validation**, not an autonomous agent. The LLM never
gets to decide a verdict; it interprets text and ranks/phrases things that a fixed set of Python
conditions then checks.

### 5.1 Retrieval (RAG)

`notes/index.py` embeds all 10 notes locally with ChromaDB's ONNX `all-MiniLM-L6-v2` model (no API
cost, no network dependency after first load) and ranks them per candidate against a query built
from the candidate's route, `route_type`, and week.

**`top_k` is 10 — i.e., every note, not a shortlist.** We tried 5 first. It broke a real case: the
query for Mumbai-Delhi ranks route-name-bearing notes above `N003`, whose text never names a route
at all — so `N003` dropped out of the top-5 shortlist and the gate never got to see it, silently
reverting a genuinely justified row to "unexplained." With a corpus this small, letting Chroma rank
everything and handing the full list to the gate is the honest fix — retrieval's job here is
demonstrating the architecture and producing an inspectable trace
(`outputs/retrieval_trace.jsonl`, one entry per candidate showing the full ranking), not narrowing
what the gate is allowed to see. At a much larger note count, `top_k` would need to shrink again,
paired with re-testing for this exact failure mode.

### 5.2 LLM note understanding — lazy, and cached per note

`notes/enrich.py` asks the LLM three things about a note's raw text, at `temperature=0`: does it
describe a cost rise (`indicates_cost_increase`), does it really apply to this dataset
(`applies_to_dataset`), and what's the verbatim evidence quote (`evidence_span`)?

- **Lazy**: a note is only enriched the first time some candidate's retrieval actually surfaces it
  — not eagerly at pipeline startup. On this 10-note/`top_k=10` corpus that still means all 10 get
  enriched on a cold run (every candidate's retrieval returns the whole corpus), but the benefit
  compounds as the corpus grows: a 1,000-note corpus with `top_k=10` would cost at most 10 calls per
  *distinct note actually retrieved*, not 1,000 calls up front.
- **Cached by `note_id`** in `outputs/notes_index.json` (committed to the repo) — this is why
  everything except a cold `run`/`cost-log` works with zero API calls.
- `applies_to` (the route) and `effective_from` (the date) are read straight from
  `context_notes.csv`'s own columns, **never** asked of the LLM — this removes a whole class of
  possible hallucination for free, since in this note set every note with an explicit start date
  already states it in its own `date` field.

### 5.3 The deterministic gate — where trust is actually won

`notes/gate.py` has **zero LLM calls**. A note justifies a candidate only if all four hold:

| # | Condition | Maps to the brief's guardrail |
|---|---|---|
| 1 | Note's `applies_to` = candidate's route, or "All Routes" | "Wrong route ... does not count" |
| 2 | Note's `[effective_from, effective_to]` overlaps the candidate's Mon–Sun week | "Wrong window ... does not count" |
| 3 | `indicates_cost_increase` AND `applies_to_dataset` both true | "a note that says costs weren't affected does not count" |
| 4 | `evidence_span` is a verbatim substring of the note's own text | nothing invented to fill a gap |

If more than one note passes (never happens here), tie-break is deterministic: route-specific
before "All Routes," then narrower window, then lexicographic `note_id` — so the verdict can't
depend on retrieval order.

**Why each distractor note fails, on purpose:**

| Note | Why it fails |
|---|---|
| N004 | Says "All Routes" but its own text says the affected routes "are not part of this dataset" — caught only by reading the prose, condition 3 |
| N005, N006, N008 | Explicitly say costs weren't significantly affected / demand stable, condition 3 |
| N007, N009 | Describe conditions *improving* / *normalizing*, condition 3 |
| N010 | Costs "absorbed ... without a rate change," condition 3 |
| N001 | Passes for weeks inside its flood window, fails condition 2 for later weeks once the road reopened (e.g. Chennai-Bangalore 2025-03-10 onward) |

**`N003`'s open-ended window** ("diesel prices rose... starting this week," no stated end date) is
the one genuinely hard case. Left unbounded it would silently justify 18 of the 27 candidates,
including the graders' own published sample row (Mumbai-Pune 2025-09-15, published as
`flagged=Yes`) — a direct contradiction, not a style choice. We derive a cutoff from the sample
itself rather than picking one freely: the last candidate the data still expects N003 to explain is
4 weeks after the note begins; the first one the sample forces back to unexplained is 19 weeks in.
We use `effective_from + 8 weeks` (`config.yaml: notes.open_ended_note_weeks`) — the midpoint of
that evidence-bound `[4, 18]` week range.

### 5.4 LLM explanation phrasing

`explain.py` turns the gate's own facts into one sentence, two modes: `llm` (default) or
`--explain-mode template` (zero LLM calls, same underlying facts — this is the structural proof
that determinism doesn't depend on the model cooperating). Has its own cache
(`outputs/explanation_cache.json`), deliberately separate from the note-enrichment cache and
intentionally cleared by `repro`/`cost-log` so the LLM genuinely regenerates prose each time.

### 5.5 Q&A (stretch goal)

```
python -m watchdog.cli ask "why did Chennai-Bangalore get pricier in March 2025?"
python -m watchdog.cli ask                                                          # interactive
```

Reads the artefacts `run` already wrote (`output.csv`, `outputs/weekly_metrics.csv`,
`data/context_notes.csv`) — does not re-run the pipeline or re-query the vector index. Same split
as the rest of the project: **code decides scope, the LLM only phrases it.**

- `parse_question` deterministically extracts a route (exact name, or a city that narrows to
  several routes — e.g. "Delhi" matches 3 routes, and the answer is told to cover each separately)
  and an optional month/year.
- `build_context` filters `weekly_metrics.csv`/`output.csv` to just those rows. No route
  recognized, or zero rows in the date window, returns a fixed message and **skips the LLM call
  entirely**.
- The LLM gets only the filtered rows (plus full text of any cited note) and phrases an answer,
  `temperature=0`.

**A real bug this caught, and how it's handled.** Live-testing an ambiguous "how is the Delhi route
doing" question, the model at different times silently dropped a route from its answer, and once
stated a wrong flagged/justified count. Free-form prose isn't a small enough surface to verify the
way a single `flagged`/`matched_note_id` pair is, so instead of trying to parse and correct the
model's sentences, **every answer gets a deterministic "Ground truth" footer** — one line per route,
computed directly from `output.csv`, letting a reader catch any slip by comparison. This is a
disclosed, known limitation of the Q&A layer, not a solved problem — only the footer is guaranteed
correct.

```
$ python -m watchdog.cli ask "why did Chennai-Bangalore get pricier in March 2025?"
The price jump in early March 2025 was explained by heavy flooding on the Chennai-Bangalore
highway... (see note N001). Subsequent weeks in March also saw higher costs, but those increases
were not linked to a specific event in the data.

Ground truth (verify the answer above against this):
- Chennai-Bangalore: 3 flagged week(s), 1 justified (N001), 2 unexplained.
```

A second, smaller bug was caught and fixed the same way after live testing: the citation-safety
check that warns when the model cites a note id not in its grounding data was comparing against too
narrow a set (only `matched_note_id` notes), so it false-flagged legitimate citations of near-miss
notes quoted inside a `reason` string the model could actually see. Fixed to check against
everything actually rendered into the prompt.

---

## 6. Guardrails — how "trust" is actually shown, not just claimed

The brief is explicit: *"a working demo isn't the bar."* Here's what backs that up, independently:

- **Hallucination**: the gate (§5.3) is the only place a verdict is decided, and has no LLM call.
  `eval/run_eval.py`'s **hallucination audit** re-derives all 9 justified rows independently from
  the committed note cache — not from what `output.csv` itself claims — and fails loudly on any
  mismatch. It also prints a citation count per note; all 7 distractors sit at zero.
- **Knowing the system is right**: `eval/run_eval.py` runs 3 separately-weighted checks:
  1. **Graders' sample check** — the 3 rows FreightTiger actually published. The only genuinely
     *external* ground truth here.
  2. **Zero-tolerance hallucination audit** (above).
  3. **Regression snapshot** against `eval/expected_verdicts.csv` — this is a transcription of this
     project's own `output.csv`, so a match proves *no drift*, not *correctness*; the eval output
     says so explicitly rather than presenting all 27 rows as one undifferentiated labeled set.
  Backed by **139 passing unit/integration tests** (`pytest`), including one test per gate
  guardrail (`tests/test_gate.py`).
- **Reproducibility**: `python -m watchdog.cli repro` runs the pipeline 3x, clearing only the
  *prose* cache between runs (not note enrichment, which is factual extraction at `temperature=0`,
  treated as stable). `route`, `week_of`, `cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`,
  `flagged`, and `matched_note_id` are identical across all 3 runs — only `reason` wording varies.
  See `REPRO.md` for the actual diff result. `temperature=0` is the only place randomness could
  enter, and `--explain-mode template` reproduces the same verdicts with **zero** LLM calls as a
  structural (not just empirical) proof.
- **Ran responsibly**: see §7 for the actual numbers — 42 calls for a full cold run, $0.00 real
  cost, an honest log rather than an unbounded-spend black box.

---

## 7. LLM usage & cost — the numbers, one full cold run

From `outputs/token_cost_log.md` (both caches cleared, entire shipment file):

| Metric | Value |
|---|---|
| Model | `openai/gpt-oss-20b` via NVIDIA NIM (OpenAI-compatible), `temperature=0` |
| Total LLM calls | **42** (10 note enrichment + 32 explanation — 27 candidates plus a few automatic retries at a larger `max_tokens` when a response got cut off mid-reasoning) |
| Total input (prompt) tokens | **12,462** |
| Total output (completion) tokens | **14,287** |
| Actual cost | **$0.00** — NVIDIA NIM free tier |
| Rough equivalent cost | **$0.0022** ($0.0004 in + $0.0019 out), using OpenRouter's published third-party rate for this open-weight model ($0.03/1M input, $0.13/1M output — no direct OpenAI list price exists since it's open-weight), checked 2026-08-19 |

`llm.py` is the single wrapper every call goes through, so `CostTracker` can log call/token counts
by purpose (`note_enrichment`, `explanation`, `qa`) — `ask` calls are tracked separately
(`purpose="qa"`) and are **not** part of this graded pipeline cost log, since they're one call per
user question, not part of the fixed `run`.

---

## 8. Design decisions & trade-offs — ours, not the brief's

The brief leaves several things undefined on purpose (a test of judgment, not an oversight).
Everything below is our call, isolated in `config.yaml`, easy to change:

| Choice | Value | Why |
|---|---|---|
| Candidacy thresholds | `vs_own_history ≥ 7%` OR `vs_similar_routes ≥ 20%` | ~97th percentile of the 728 observed weekly changes — picks out clear outliers without dragging in ordinary noise. No sensitivity sweep was run; this is a judgment call. `python -m eval.threshold_analysis` recomputes both percentiles from the data directly (currently 97.2th/97.4th) rather than taking this row's word for it |
| Near-miss window | ±14 days | How close a *failing* note has to be, by date, to be named in an unexplained row's `reason` as "the closest note that didn't hold up" |
| N003's open-ended window | `effective_from + 8 weeks` | Derived from the sample's own two anchor rows — see §5.3 |
| CSV quoting | `csv.QUOTE_MINIMAL` on write | The supplied sample isn't valid CSV as-is; we write one that round-trips through pandas |
| Retrieval `top_k` | 10 (every note) | A smaller shortlist measurably dropped a real match — see §5.1 |
| LLM model | `openai/gpt-oss-20b` on NVIDIA NIM | Verified working, open-weight, free tier — matches the brief's "prefer open-source/free-tier" |

---

## 9. Limitations (disclosed, not hidden)

- The 7%/20% candidacy thresholds and the 14-day near-miss window are judgment calls, not derived
  from a labeled ground truth of "real" anomalies.
- `Short` and `Long` route types have only one peer route each, so `vs_similar_routes` there is a
  single-route comparison, not a true average.
- The 8-week open-ended-note window is anchored to this dataset's one open-ended note and the
  sample's two anchor rows — a different dataset would need its own re-derivation, not a blind reuse.
- `top_k=10` on a 10-note corpus demonstrates ranking, not filtering; a larger corpus would need
  `top_k` re-tuned and re-tested against the same dropped-match failure mode.
- Weekly figures rest on 3–5 shipments per route-week — not a lot of signal per data point.
- `temperature=0` reduces variance on the hosted API but doesn't guarantee bit-identical output
  run-to-run. Confirmed directly: a genuinely ambiguous field (`N009`'s `effective_to`, which names
  no date in its own text) returned different values across two cold runs before the prompt was
  tightened. It never changed a `flagged`/`matched_note_id` result (N009 fails the gate on a
  different condition regardless), but the enrichment cache isn't provably deterministic the way
  `notes/gate.py` is.
- The Q&A layer's prose can still be locally wrong (§5.4) — only its ground-truth footer is
  guaranteed correct, by construction, not by the model behaving.

---

## 10. Further improvements, if this went past 24 hours

- Replace the percentile-based candidacy thresholds with ones calibrated against a real labeled
  anomaly set, once one exists, instead of a heuristic sweep of this one dataset.
- Move to a cross-encoder reranker (or just shrink `top_k` again with re-testing) once the note
  corpus grows past the point where "rank everything" stays cheap and safe.
- Ask the LLM for structured (function-calling / JSON-schema) output in `enrich.py`/`explain.py`
  instead of prompt-based parsing, to cut the truncation-retry overhead visible in the call count.
- Batch note-enrichment calls instead of one request per note, to cut round-trips as the corpus
  grows.
- Extend `ask.py`'s regex-based route/date parsing to handle fuzzier phrasing, or add an
  LLM-based slot-extraction step with the current deterministic parser kept as a verified fallback.
- Add CI (e.g. GitHub Actions) running `pytest` + `eval/run_eval.py` on every change, so a
  regression in the gate invariant fails a build, not just a manual review.
- A minimal read-only dashboard over `output.csv`/`weekly_metrics.csv` for a non-technical
  stakeholder, instead of requiring someone to open a CSV.
- Alerting (e.g. Slack/email) when a fresh run produces a new unexplained spike, rather than
  requiring someone to diff `output.csv` by hand.

---

## 11. Deliverables checklist (per "What you need to hand in")

| Asked for | Where |
|---|---|
| Working code | `src/watchdog/` |
| Output CSV matching the sample format exactly | `output.csv` |
| Short README: how it works, how built, how self-verified | this file |
| Reproducibility check across 3 runs | `REPRO.md` |
| Token/cost log for one full run | `outputs/token_cost_log.md` |
| 10-min walkthrough | `WALKTHROUGH.md` |

## 12. Self-assessment against "How we'll evaluate you"

| Criterion | Where this is addressed |
|---|---|
| Does the core logic work? | §3 — every core requirement mapped to code + how it was verified |
| Did you use AI meaningfully? | §5 — RAG retrieval with a documented failure mode we fixed, LLM note-understanding + phrasing, both cached and cost-tracked |
| Can we trust it? | §6 — deterministic gate, independent hallucination audit, distractor notes at zero citations |
| Does it hold up on repeat? | §6 — 3 identical runs on every graded column, plus a zero-LLM structural proof |
| Did you run it responsibly? | §7 — 42 calls, $0.00 actual cost, full breakdown |
| Is the code clean? | One module per pipeline stage (§2), 139 tests, `config.yaml` isolates every judgment call from the logic |
| Can you explain your work? | This document + `WALKTHROUGH.md` |

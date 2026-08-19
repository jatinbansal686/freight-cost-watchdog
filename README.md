# Freight Cost Watchdog

A FreightTiger 24-hour case study submission: an assistant that watches per-route freight cost,
flags weeks where it's rising in a way that looks out of the ordinary, checks a small set of
context notes for a real reason before saying anything is "justified," and refuses to invent one
when none exists.

**`output.csv` at the repo root is the submission.**

## 1. What it does, how to run it

```
pip install -r requirements.txt
cp .env.example .env        # fill in NVIDIA_API_KEY
python -m watchdog.cli run                          # full run, writes output.csv
python -m watchdog.cli run --explain-mode template   # zero LLM calls, identical verdicts/numbers
python -m watchdog.cli run --no-cache                # ignore caches, force a cold run
python -m watchdog.cli repro                         # 3x reproducibility check -> REPRO.md
python -m watchdog.cli cost-log                      # one full cold run -> outputs/token_cost_log.md
python -m eval.run_eval                              # labelled-set + hallucination audit
pytest                                                # unit/integration tests (no API key needed)
```

Everything except `watchdog.cli run` with a cleared cache and `cost-log` works with **no API key
at all**: note enrichment is cached at `outputs/notes_index.json` (committed), and
`--explain-mode template` produces `output.csv` with identical verdicts and numbers from pure code.

Result on the supplied data: **27 candidate route-weeks, 9 justified, 18 unexplained.** Only
`N001`, `N002` and `N003` are ever cited as a match; the other seven notes are cited zero times
(verified by `eval/run_eval.py`).

## 2. Architecture

```
shipment_records.csv
        |
        v
[1] weekly aggregation      Mon-Sun weeks, pooled cost per tonne-km            (weekly.py)
        |
        v
[2] baselines                trailing 8-week own history + same-week peer avg  (baselines.py)
        |
        v
[3] suspicious candidates    cost rising and out of the ordinary               (detect.py)
        |
        v
[4] RAG retrieval            ChromaDB over context_notes.csv     <- LLM/AI layer (notes/index.py)
        |
        v
[5] LLM note understanding   extract window / cost direction / evidence quote  (notes/enrich.py)
        |
        v
[6] DETERMINISTIC GATE       route AND window AND raises-cost AND verbatim evidence (notes/gate.py)
        |
        +-- valid note found --> final result = "No (justified)" + matched_note_id
        +-- none found --------> final result = "Yes"            + blank matched_note_id
        |
        v
[7] plain-English reason     grounded in the gate's own facts only             (explain.py)
        |
        v
    output.csv                                                                (report.py)
```

**The LLM proposes, code decides.** The LLM's jobs are: understand a note's prose (does it
describe a cost rise? does it really apply to routes in this dataset? what's the verbatim
evidence?), rank notes for a candidate via retrieval, and phrase the final sentence. It never sets
`flagged` or `matched_note_id` -- `notes/gate.py` is the only place those are decided, and it is
plain deterministic Python with no LLM call in it.

### Suspicious candidate vs. final result

These are deliberately different things, kept distinct in the code and here:

- **Suspicious candidate** -- a route-week whose cost rose unusually (`detect.py`). Internal,
  pre-investigation. 27 of the 728 route-weeks qualify.
- **Final result** -- what ships in `output.csv`. `No (justified)` only if a note survives all
  four gate conditions; `Yes` otherwise. A candidate is *investigated*, not flagged --
  `flagged` is the answer, not the question.

## 3. Baselines, exactly as specified

- **vs. own history**: the route's trailing 8-week rolling average `cost_per_tonne_km`, using only
  weeks strictly before the current week (no look-ahead). **If fewer than 8 prior weeks exist, we
  use all prior weeks available and do not pad or extrapolate** -- the exact count used is in the
  `baseline_weeks_used` column of `outputs/weekly_metrics.csv`. A route's very first week has zero
  prior weeks, so `vs_own_history` is blank there.
- **vs. similar routes**: the unweighted average `cost_per_tonne_km`, in that same week, across the
  *other* routes sharing the same `route_type`. The route itself is excluded from its own peer
  average (`peer_count` in `weekly_metrics.csv`). Note: `Short` = {Delhi-Jaipur, Mumbai-Pune} and
  `Long` = {Delhi-Chennai, Mumbai-Delhi} each have exactly **one** peer route; `Medium` has three.

`cost_per_tonne_km` itself is **pooled**: `sum(freight_cost_inr) / sum(quantity_tonnes *
distance_km)` across all shipments in the route-week, not the mean of each shipment's own ratio.
Verified against all three rows of the supplied `sample_output_format_v2.csv` to the exact digit.

## 4. Our implementation choices (not FreightTiger's)

The assignment leaves several things undefined on purpose. Everything below is *our* call, kept
out of the graded logic and easy to change in `config.yaml`.

| Choice | Value | Why |
|---|---|---|
| Candidacy thresholds | `vs_own_history >= 7%` OR `vs_similar_routes >= 20%` | Roughly the 97th percentile of the 728 observed weekly changes -- picks out clear outliers without dragging in ordinary noise. No sensitivity sweep was run; this is a judgment call. |
| Near-miss window | +/- 14 days | How close a *failing* note has to be, by date, to be named in an unexplained row's `reason` as "the closest note that didn't hold up," rather than saying nothing was found at all. |
| CSV quoting | `csv.QUOTE_MINIMAL` on write | The supplied `sample_output_format_v2.csv` is not valid CSV (see below); we write a file that actually round-trips through `pandas`. |
| `_v2` file is binding | `sample_output_format_v2.csv`, not `sample_output_format.csv` (named in the brief's prose) | It's the file we were given, and the only one that contains `matched_note_id`, which the brief explicitly names as part of the contract. |
| Retrieval `top_k` | 10 (i.e. every note) | See the retrieval section below -- a smaller top_k measurably dropped a real match. |
| LLM model | `openai/gpt-oss-20b` on NVIDIA NIM | Verified working against the API key we were given for this submission; open-weight, free tier. |

### The supplied sample format file doesn't parse

`sample_output_format_v2.csv` row 4's `reason` field contains an unquoted comma
(`... (N006, 2025-09-22) ...`) against an 8-column header with zero quote characters anywhere in
the file -- `pd.read_csv` raises a `ParserError` on it. `tests/test_output_format.py` reads just
the header line directly rather than asking pandas to parse the whole file, and the three example
rows are hand-transcribed into `tests/test_sample_rows.py`'s fixture for the same reason. Our own
`output.csv` is written with `QUOTE_MINIMAL`, which does not have this problem.

### The open-ended note (N003) window

N003 ("Diesel prices rose nationwide **starting this week**...") never states an end date. Left
unbounded, it would silently justify **every** later candidate on every route from 2025-05-05
onward -- 18 of the 27 rows, including the sample's own **Mumbai-Pune 2025-09-15**, which the
graders publish as `flagged=Yes` with `matched_note_id` blank and a `reason` that discusses a
*different* note (N006) and rejects it. Treating N003 as unbounded would output
`No (justified), N003` there instead -- a direct contradiction of the graded sample, not a style
choice.

So a cutoff is required, and it's **derived from the sample rather than picked freely**: the last
candidate the data itself still expects N003 to explain is **Mumbai-Delhi 2025-06-02** (4 weeks
after the note begins, and it is in fact justified by N003 in our output); the first candidate the
sample explicitly forces back to unexplained is **Mumbai-Pune 2025-09-15** (19 weeks in). We use
`effective_from + 8 weeks` (`config.yaml: notes.open_ended_note_weeks`) -- the midpoint of that
evidence-bound range `[4, 18]` weeks, not an arbitrary guess.

## 5. The gate: where trust is won

A note may justify a candidate only if **all four** hold (`notes/gate.py`), each mapped to a
guardrail the assignment names verbatim:

| # | Condition | Assignment guardrail |
|---|---|---|
| 1 | Note's `applies_to` equals the candidate's route, or is "All Routes" | "**Wrong route** ... does not count" |
| 2 | Note's `[effective_from, effective_to]` overlaps the candidate's Mon-Sun week | "**Wrong window** ... does not count" |
| 3 | `indicates_cost_increase` AND `applies_to_dataset` are both true | "a note that **says costs weren't affected** does not count" |
| 4 | `evidence_span` is a verbatim substring of the note's own text | nothing invented to fill a gap |

If more than one note passes (never happens in this dataset), the tie-break is deterministic:
route-specific before "All Routes", then the narrower window, then lexicographic `note_id` -- so
the verdict cannot depend on retrieval order.

`applies_to` (the route) is read straight from `context_notes.csv`, never asked of the LLM.
`applies_to_dataset` is a separate, LLM-derived flag that exists specifically to catch **N004**:
its `applies_to` column literally says "All Routes", but the note's own prose says the affected
routes "are not part of this dataset" -- only reading the text catches that contradiction.
`effective_from` is likewise always the note's own `date` column, never LLM-derived, because in
this 10-note set every note that states an explicit start date states the same date already in
`date` -- this removes one whole class of possible hallucination for free.

The seven distractor notes and why each fails, on purpose:

| Note | Why it fails the gate |
|---|---|
| N004 | Says the affected routes "are not part of this dataset" -- condition 3 |
| N005, N006, N008 | Explicitly say costs were *not* significantly affected / demand stable, no disruption -- condition 3 |
| N007, N009 | Describe conditions *improving* / *returning to normal* -- condition 3 |
| N010 | Costs "absorbed ... without a rate change" -- condition 3 |
| N001 | Passes for the weeks its flood window covers, but fails condition 2 for later weeks (e.g. Chennai-Bangalore 2025-03-10, 03-17) once the road reopened |

## 6. Retrieval (RAG)

`notes/index.py` embeds the 10 notes with ChromaDB's local ONNX `all-MiniLM-L6-v2` model (no API
cost) and ranks them against a query built from the candidate's route, `route_type`, and week.
`outputs/retrieval_trace.jsonl` records, per candidate, the full ranking and the gate's decision on
each retrieved note -- the artefact that shows retrieval actually working, not just a black box.

**`top_k` is 10 (i.e. every note), not a smaller shortlist.** We tried 5 first, matching the
architecture described in the original plan. It broke a real case: a query built from
`"route Mumbai-Delhi, route type Long, week of 2025-06-02"` ranks route-name-bearing notes (N005,
N007, a wrong-route note) above N003, whose text never mentions a route name at all -- so N003
dropped out of the top-5 shortlist and the gate never got to see it, silently reverting a genuinely
justified row to "unexplained." With a corpus this small (10 notes), the honest fix is to let
Chroma rank everything and hand the full ranked list to the gate, which is still the sole decision
authority -- retrieval's job here is demonstrating the architecture and producing the trace, not
narrowing what the gate is allowed to consider. If Chroma or its embedding model fails to load
(e.g. no network on first run), `notes/index.py` falls back to an unranked all-notes list with the
same effect on the gate.

## 7. How I convinced myself it was right

- **The three sample rows reproduce exactly**, to the digit, on `cost_per_tonne_km`,
  `vs_own_history`, `vs_similar_routes`, `flagged` and `matched_note_id`
  (`tests/test_sample_rows.py`).
- **One test per gate guardrail** (`tests/test_gate.py`): wrong route rejected, wrong window
  rejected (using an N001-shaped note against Chennai-Bangalore 2025-03-10), "costs not affected"
  rejected, improvement/normalisation rejected, out-of-dataset rejected (the N004 shape), a valid
  match accepted, no note passing yields a blank `matched_note_id`, non-verbatim evidence rejected,
  and the N003 open-ended window tested both active and expired.
- **`eval/run_eval.py`** scores the full output against a hand-labelled `eval/expected_verdicts.csv`
  for all 27 candidates, then runs a **zero-tolerance hallucination audit**: for every
  `No (justified)` row it independently re-derives the gate check from the committed note cache
  (not from what the pipeline itself claimed) and fails the run on any mismatch. It also prints a
  citation count per note -- the seven distractors sit at zero.
- **What I did *not* verify**: the 7%/20% thresholds are a judgment call, not derived from a
  labelled ground truth of "real" anomalies. `Short` and `Long` peer groups have only one peer
  route each, so `vs_similar_routes` for those routes is noisier than for `Medium`. Weekly figures
  rest on only 3-5 shipments each.

## 8. Reproducibility

`python -m watchdog.cli repro` runs the pipeline 3x, clearing the prose cache (not the note
enrichment cache -- that's deterministic factual extraction, not style) between runs so the LLM
genuinely regenerates every `reason` string each time. `route`, `week_of`, `cost_per_tonne_km`,
`vs_own_history`, `vs_similar_routes`, `flagged` and `matched_note_id` are identical across all 3
runs; only `reason` wording can vary. See `REPRO.md` for the actual result. The only place
randomness could enter is the LLM, and `temperature=0` there; `--explain-mode template` produces
the same verdicts and numbers with **zero** LLM calls, which is the structural proof that
determinism doesn't depend on the model cooperating.

## 9. Cost

See `outputs/token_cost_log.md` for the full breakdown from one cold run (both caches cleared) over
the entire shipment file: **10 notes to enrich + 27 candidates to explain = 37 "logical" LLM
requests.** `llm.py` retries a call at a larger `max_tokens` if the model's response got cut off
mid-reasoning (see `notes/enrich.py` / `explain.py` docstrings), so the *actual* call count in the
log can run a little above 37 -- the log states the true number for that run, split by purpose.
Actual cost either way: **$0.00** on the NVIDIA NIM free tier. `openai/gpt-oss-20b` has no direct
OpenAI list price (it's open-weight); the log's "rough equivalent" uses the published third-party
rate from [OpenRouter](https://openrouter.ai/openai/gpt-oss-20b) -- $0.03/1M input, $0.13/1M output
tokens, checked 2026-08-19 -- as a stand-in, cited in `cli.py`.

## 10. Limitations

- The 7%/20% candidacy thresholds and the 14-day near-miss window are our judgment calls, not
  derived from labelled ground truth.
- `Short` and `Long` route types have only one peer route each in this dataset, so their
  `vs_similar_routes` figure is a single-route comparison, not a true average.
- The open-ended-note window (8 weeks) is anchored to this dataset's one open-ended note and the
  sample's own two anchor rows; a different dataset with a different open-ended note would need
  its own check, not a blind reuse of 8 weeks.
- Retrieval `top_k=10` on a 10-note corpus means the vector search demonstrates ranking rather than
  filtering; at a larger note count this would need re-tuning (and re-testing against dropped
  matches the way we caught this one).
- Weekly figures rest on 3-5 shipments per route-week -- not a lot of signal per data point.

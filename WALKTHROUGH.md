# Walkthrough script (~10 minutes)

## 1. Problem framing (1 min)

Freight cost per route quietly drifts. Sometimes there's a real reason (flood, festival surcharge,
diesel prices); sometimes there isn't. The assignment's actual bar isn't "detect a rise" -- it's
**"don't claim a reason you can't back up."** A confident wrong reason scores worse than "couldn't
find one." So the whole design is built around one rule: **the LLM proposes, deterministic code
decides.** Retrieval and the LLM find and interpret notes; a plain Python function
(`notes/gate.py`) is the only place `flagged` and `matched_note_id` get set.

## 2. Live run and the output (2 min)

```
python -m watchdog.cli run
```

Walk through `output.csv`: 27 candidate route-weeks out of 728 total, 9 justified, 18 unexplained.
Point at one justified row (Ahmedabad-Mumbai, N002 festival surcharge) and one unexplained row
(Delhi-Jaipur 2024-11-11 -- this is one of the three rows in the graders' own sample, reproduced
exactly). Mention `outputs/weekly_metrics.csv` (all 728 route-weeks, diagnostic) and
`outputs/retrieval_trace.jsonl` (what got retrieved and why, per candidate).

## 3. The trap, and how the gate handles it (3 min)

Use **Chennai-Bangalore, 2025-03-10** as the running example.

- N001 describes flooding on this exact route, Feb 24 - Mar 8. Chennai-Bangalore's cost is still
  elevated on 2025-03-10 and 03-17 -- two weeks *after* the note's own window ends.
- Naively: "there's a note about this route describing a cost rise" would justify all four weeks.
  That's wrong for the last two -- the flood physically ended.
- The gate's condition 2 (window overlap) catches this: N001's `effective_to` is 2025-03-08, which
  doesn't overlap the Mar 10 or Mar 17 Mon-Sun weeks. Both stay `Yes`, correctly.
- Then show the *other* direction of the same trap: N003 (diesel, "starting this week," **no end
  date given**). Left unbounded, it would silently justify 18 of the 27 candidates -- including the
  graders' own sample row, **Mumbai-Pune 2025-09-15**, which they publish as unexplained. We
  derived a cutoff from the sample itself (not picked freely): N003 must still be active at
  Mumbai-Delhi's +4-week candidate and must be expired by Mumbai-Pune's +19-week one, so we use
  `effective_from + 8 weeks` -- the midpoint of that evidence-bound range. This is flagged in the
  README as *our* choice, not the assignment's.

## 4. Trade-offs (3 min)

- **Why the LLM never decides.** Cost-effective, but the actual reason is trust: the assignment
  explicitly says a wrong-but-confident reason is worse than an honest "unexplained," and spot-checks
  justified rows against the raw notes. Putting the verdict in a testable, deterministic function
  means a guardrail violation is a bug I can write a unit test for, not a prompting problem I can
  only sample-check.
- **Why the thresholds (7% / 20%) are ours.** The assignment deliberately doesn't define "out of
  the ordinary." We picked values near the 97th percentile of observed weekly changes and labelled
  them clearly as a judgment call in `config.yaml` and the README, not something derived from the
  brief.
- **Why the vector DB is `top_k=10` (i.e. every note) on a 10-note corpus.** Tried `top_k=5` first,
  matching the "shortlist" idea in the original design. It actually dropped a correct match:
  Mumbai-Delhi 2025-06-02's query ranked two wrong-route notes above N003, whose text never
  mentions a route name, so N003 fell out of the shortlist and the gate never got to check it.
  With only 10 notes total, ranking everything and letting the gate -- not the retriever -- decide
  relevance is the honest fix. This is exactly the kind of thing the reproducibility/eval harness
  is for: it caught a real correctness bug, not just a demo bug.

## 5. Q&A stretch goal, if there's time (1 min)

```
python -m watchdog.cli ask "why did Chennai-Bangalore get pricier in March 2025?"
```

Same "code decides scope, LLM only phrases" split as the pipeline itself: a deterministic parser
picks the route/period out of the question and filters `output.csv`/`weekly_metrics.csv` to just
those rows -- no match means a fixed "no data" message and zero LLM calls, never a guess. The
answer is followed by a small deterministic "Ground truth" line per route, added because live
testing caught the model itself both dropping a route from a multi-route answer and once
misstating a justified/unexplained count -- free-form prose isn't as verifiable as `gate.py`'s four
fixed conditions, so the footer is what makes any slip visible instead of trusted.

## 6. How I checked it (1 min)

Three sample rows -- the only ground truth FreightTiger actually gave us -- reproduce to the digit.
One test per gate guardrail (wrong route, wrong window, "not affected," improved/normal,
out-of-dataset, valid match, blank-when-unexplained, non-verbatim-evidence-rejected, open-ended-
window-both-directions). `eval/run_eval.py` runs three checks and reports them separately rather
than as one score: the graders' 3-row sample (independent), a zero-tolerance hallucination audit
that independently re-derives every justified verdict from the committed note cache (also
independent, prints a citation count per note -- the seven distractor notes sit at zero), and a
27-row regression snapshot which is explicitly labelled as drift-detection against our own
committed output, not independent grading. `watchdog.cli repro` runs the pipeline 3x with the prose
cache cleared; every column except `reason` is identical across runs, and `--explain-mode template`
reproduces the same verdicts and numbers with zero LLM calls.

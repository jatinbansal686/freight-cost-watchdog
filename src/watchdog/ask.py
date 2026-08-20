"""Q&A layer over the pipeline's own results (stretch goal from the brief).

Same division of labor as the rest of the pipeline: code decides which rows are in scope for a
question (deterministic route/date parsing + filtering against the already-written output.csv /
outputs/weekly_metrics.csv), the LLM only turns those rows into a plain-English sentence. If the
question doesn't resolve to a known route, or a date window has no data, we say so and skip the
LLM call entirely rather than let the model guess -- the same "no supporting data -> say so"
guardrail as notes/gate.py, applied here to Q&A instead of verdicts.

This reads the artefacts `watchdog.cli run` already wrote; it does not re-run the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from watchdog.llm import LLMClient

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Words/symbols that plausibly join two city names into one route reference -- "Delhi-Chennai",
# "Delhi to Chennai", "Chennai and Delhi". Deliberately a fixed, REQUIRED connector list -- two
# earlier approaches were tried and rejected because both let a false match through:
#   1. A character-distance window (e.g. "within 20 chars") matches "Delhi to Chennai and Mumbai
#      to Pune" as also naming a spurious Mumbai-Delhi route, since "delhi" and "mumbai" end up
#      within 20 characters of each other despite belonging to two unrelated route mentions.
#   2. Making this connector *optional* (so bare whitespace adjacency also matched) reopened a
#      narrower version of the same bug: "compare Mumbai Delhi Chennai trends" (three cities, no
#      connector word anywhere) matched both Mumbai-Delhi and Delhi-Chennai off the shared "Delhi".
# The connector must actually be present -- bare adjacency alone is not enough.
_ROUTE_CONNECTOR = r"(?:-|to|and|&|vs\.?|versus)"

SYSTEM_PROMPT = """You answer questions about a freight-cost analysis using ONLY the data given \
below -- never state a number, route, week, or note that isn't in it. If the data doesn't answer \
the question, say so plainly instead of guessing. Plain English, and cite a note id (e.g. N002) \
only when you are directly referencing one from the data. If the data lists more than one route \
under "Route summary", you MUST give each one its own sentence covering its finding, even if that \
route has zero flagged weeks -- do not silently drop a route just because it has less to say."""


def known_routes(weekly_df: pd.DataFrame) -> list[str]:
    return sorted(weekly_df["route"].unique().tolist())


@dataclass
class ParsedQuery:
    routes: list[str]
    year: int | None
    month: int | None
    ambiguous: bool
    months_mentioned: list[int] = field(default_factory=list)
    ambiguous_month: bool = False


def _route_named(question_lower: str, city_a: str, city_b: str) -> bool:
    """True if a route's two cities appear directly joined by a connector, in either order --
    covers "Delhi-Chennai", "Delhi to Chennai", "from Chennai to Delhi", "Chennai and Delhi",
    not just the exact hyphenated spelling. The two city names must be adjacent (only a connector
    token or whitespace between them) so this can't bridge across an unrelated third city."""
    a, b = re.escape(city_a.lower()), re.escape(city_b.lower())
    connector = rf"\s*{_ROUTE_CONNECTOR}\s*"
    pattern = rf"\b{a}\b{connector}\b{b}\b|\b{b}\b{connector}\b{a}\b"
    return re.search(pattern, question_lower) is not None


def parse_question(question: str, routes: list[str]) -> ParsedQuery:
    q = question.lower()

    exact = [r for r in routes if _route_named(q, *r.split("-"))]
    ambiguous = False
    if exact:
        matched_routes = exact
    else:
        cities = {c for r in routes for c in r.split("-")}
        hit_cities = [c for c in cities if re.search(rf"\b{re.escape(c.lower())}\b", q)]
        matched_routes = sorted({r for r in routes if any(c in r.split("-") for c in hit_cities)})
        ambiguous = len(matched_routes) > 1

    months_mentioned: list[int] = []
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q) and num not in months_mentioned:
            months_mentioned.append(num)
    months_mentioned.sort()
    ambiguous_month = len(months_mentioned) > 1
    month = months_mentioned[0] if len(months_mentioned) == 1 else None

    year_match = re.search(r"\b(20\d{2})\b", q)
    year = int(year_match.group(1)) if year_match else None

    return ParsedQuery(
        routes=matched_routes, year=year, month=month, ambiguous=ambiguous,
        months_mentioned=months_mentioned, ambiguous_month=ambiguous_month,
    )


@dataclass
class QAContext:
    routes: list[str]
    ambiguous: bool
    summary_lines: list[str] = field(default_factory=list)
    candidate_rows: list[dict] = field(default_factory=list)
    windowed_rows: list[dict] = field(default_factory=list)
    note_texts: dict[str, str] = field(default_factory=dict)
    summary_by_route: dict[str, str] = field(default_factory=dict)
    no_data_reason: str | None = None
    ambiguous_month: bool = False
    months_mentioned: list[int] = field(default_factory=list)


def _note_text(notes_df: pd.DataFrame, note_id: str) -> str | None:
    hit = notes_df[notes_df["note_id"] == note_id]
    return None if hit.empty else str(hit.iloc[0]["note"])


def build_context(
    parsed: ParsedQuery,
    weekly_df: pd.DataFrame,
    output_df: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> QAContext:
    if not parsed.routes:
        return QAContext(routes=[], ambiguous=False, no_data_reason="no_route")

    route_weekly = weekly_df[weekly_df["route"].isin(parsed.routes)].copy()
    route_weekly["week_of_dt"] = pd.to_datetime(route_weekly["week_of"])

    # When multiple months were named ("Feb or March"), scope to the UNION of those months, not
    # "no month filter at all" -- the latter would silently hand back the whole year (or the
    # route's entire history) while the disclosure note below tells the LLM the data "covers all
    # of them," which is only true if "them" means the named months, not every month that exists.
    if parsed.ambiguous_month:
        month_filter: set[int] | None = set(parsed.months_mentioned)
    elif parsed.month is not None:
        month_filter = {parsed.month}
    else:
        month_filter = None

    windowed_rows: list[dict] = []
    has_window = month_filter is not None or parsed.year is not None
    if has_window:
        mask = pd.Series(True, index=route_weekly.index)
        if parsed.year is not None:
            mask &= route_weekly["week_of_dt"].dt.year == parsed.year
        if month_filter is not None:
            mask &= route_weekly["week_of_dt"].dt.month.isin(month_filter)
        windowed = route_weekly[mask]
        if windowed.empty:
            return QAContext(routes=parsed.routes, ambiguous=parsed.ambiguous, no_data_reason="no_window_data")
        windowed_rows = windowed.drop(columns=["week_of_dt"]).to_dict("records")

    candidate_rows = output_df[output_df["route"].isin(parsed.routes)].to_dict("records")
    if has_window:
        candidate_rows = [
            r for r in candidate_rows
            if (parsed.year is None or pd.Timestamp(r["week_of"]).year == parsed.year)
            and (month_filter is None or pd.Timestamp(r["week_of"]).month in month_filter)
        ]

    note_texts: dict[str, str] = {}
    for row in candidate_rows:
        nid = row.get("matched_note_id")
        if nid:
            text = _note_text(notes_df, nid)
            if text:
                note_texts[nid] = text

    summary_by_route: dict[str, str] = {}
    for route in parsed.routes:
        rw = route_weekly[route_weekly["route"] == route]
        if rw.empty:
            continue
        summary_by_route[route] = (
            f"{route}: {len(rw)} weeks tracked ({rw['week_of'].min()} to {rw['week_of'].max()}), "
            f"cost_per_tonne_km ranged {rw['cost_per_tonne_km'].min():.2f}-{rw['cost_per_tonne_km'].max():.2f}, "
            f"{int(rw['is_candidate'].sum())} week(s) flagged as candidates."
        )

    return QAContext(
        routes=parsed.routes,
        ambiguous=parsed.ambiguous,
        summary_lines=list(summary_by_route.values()),
        candidate_rows=candidate_rows,
        windowed_rows=windowed_rows,
        note_texts=note_texts,
        summary_by_route=summary_by_route,
        ambiguous_month=parsed.ambiguous_month,
        months_mentioned=parsed.months_mentioned,
    )


def _fmt_pct(x) -> str:
    return "" if pd.isna(x) else f"{float(x) * 100:+.1f}%"


def _render_context_block(ctx: QAContext) -> str:
    lines: list[str] = []
    if ctx.ambiguous:
        lines.append(
            f"Note: the question matched multiple routes by city name ({', '.join(ctx.routes)}) "
            "-- answer for each separately and say which route each fact belongs to."
        )
    if ctx.ambiguous_month:
        month_names = ", ".join(MONTH_NAMES[m] for m in ctx.months_mentioned)
        lines.append(
            f"Note: the question named multiple months ({month_names}) -- the data below is "
            f"scoped to exactly those {len(ctx.months_mentioned)} months, not to a single one "
            "picked arbitrarily. Answer for each month separately and say which month each fact "
            "belongs to; do not silently collapse them into one."
        )
    lines.append("Route summary:")
    lines.extend(f"- {line}" for line in ctx.summary_lines)

    lines.append("")
    lines.append("Flagged/candidate weeks for these routes (this is the pipeline's actual verdict data):")
    if ctx.candidate_rows:
        for r in ctx.candidate_rows:
            lines.append(
                f"- {r['route']} week of {r['week_of']}: flagged={r['flagged']}, "
                f"matched_note_id={r.get('matched_note_id') or 'none'}, reason: {r['reason']}"
            )
    else:
        lines.append("- none of these routes have any flagged/candidate weeks in this period")

    if ctx.windowed_rows:
        lines.append("")
        lines.append("Raw weekly figures matching the question's time window:")
        for r in ctx.windowed_rows:
            lines.append(
                f"- {r['route']} week of {r['week_of']}: cost_per_tonne_km={r['cost_per_tonne_km']:.2f}, "
                f"vs_own_history={_fmt_pct(r.get('vs_own_history'))}, "
                f"vs_similar_routes={_fmt_pct(r.get('vs_similar_routes'))}, "
                f"is_candidate={r['is_candidate']}"
            )

    if ctx.note_texts:
        lines.append("")
        lines.append("Full text of the note(s) cited above:")
        for nid, text in ctx.note_texts.items():
            lines.append(f"- {nid}: {text}")

    return "\n".join(lines)


def _verdict_line(route: str, candidate_rows: list[dict]) -> str:
    rows = [r for r in candidate_rows if r["route"] == route]
    if not rows:
        return f"{route}: no flagged/candidate weeks in this period."
    justified = [r for r in rows if str(r["flagged"]).startswith("No")]
    unexplained = [r for r in rows if r not in justified]
    parts = [f"{len(rows)} flagged week(s)"]
    if justified:
        note_ids = sorted({r["matched_note_id"] for r in justified if r.get("matched_note_id")})
        parts.append(f"{len(justified)} justified ({', '.join(note_ids)})")
    if unexplained:
        parts.append(f"{len(unexplained)} unexplained")
    return f"{route}: " + ", ".join(parts) + "."


def answer_question(question: str, ctx: QAContext, client: LLMClient, all_routes: list[str], max_tokens: int) -> str:
    if ctx.no_data_reason == "no_route":
        return "I couldn't match that question to a known route. Known routes: " + ", ".join(all_routes) + "."
    if ctx.no_data_reason == "no_window_data":
        return f"I don't have any data for {', '.join(ctx.routes)} in that time period."

    context_block = _render_context_block(ctx)
    user_prompt = f"DATA:\n{context_block}\n\nQUESTION: {question}"
    answer = client.complete(SYSTEM_PROMPT, user_prompt, purpose="qa", max_tokens=max_tokens)
    if answer is None:
        return "(LLM call unavailable -- here is the grounded data directly instead)\n\n" + context_block

    cited = set(re.findall(r"\bN0\d{2}\b", answer))
    unknown = cited - set(ctx.note_texts)
    if unknown:
        answer += f"\n\n[warning: response cited note id(s) {sorted(unknown)} that weren't in the grounding data -- treat with caution]"

    # Defense in depth, same reasoning as notes/gate.py re-checking evidence_ok itself rather than
    # trusting enrich.py: in testing, this model stated "Delhi-Jaipur: 3 flagged weeks, all
    # unexplained" when one of those three was in fact justified by N003 -- a real factual slip in
    # the prose, not caught by the note-id check above (it cited no note at all for that route).
    # Free-form prose can't be verified the way gate.py's 4 fixed conditions can, so instead of
    # trying to parse and correct the model's sentence, always print the actual per-route verdict
    # counts underneath it -- deterministic, from candidate_rows, so any slip is visible immediately
    # rather than trusted silently.
    if ctx.routes:
        ground_truth = "\n".join(f"- {_verdict_line(r, ctx.candidate_rows)}" for r in ctx.routes)
        answer += "\n\nGround truth (verify the answer above against this):\n" + ground_truth

    return answer

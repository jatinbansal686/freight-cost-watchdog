"""Tests for the deterministic retrieval half of ask.py -- route/date parsing and row filtering.
The LLM-phrasing half (answer_question's call to client.complete) is intentionally not tested here,
same rationale as explain.py: free-form prose isn't a thing to assert equality on. What matters
for trust is that only in-scope rows ever reach the LLM, which is what these tests check."""
from watchdog.ask import _render_context_block, _route_named, build_context, known_routes, parse_question


def test_known_routes_are_the_seven_in_the_dataset(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    assert routes == sorted(routes)
    assert "Chennai-Bangalore" in routes
    assert len(routes) == 7


def test_exact_route_name_matched(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("why did Chennai-Bangalore get pricier in March", routes)
    assert parsed.routes == ["Chennai-Bangalore"]
    assert parsed.month == 3
    assert parsed.ambiguous is False


def test_city_name_fallback_matches_route(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("what happened on the Bangalore route", routes)
    assert parsed.routes == ["Chennai-Bangalore"]


def test_ambiguous_city_matches_multiple_routes(weekly_metrics_df):
    """'Delhi' appears in Delhi-Chennai, Delhi-Jaipur and Mumbai-Delhi -- all three should come
    back, flagged ambiguous, rather than silently picking one."""
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("how is the Delhi route doing", routes)
    assert set(parsed.routes) == {"Delhi-Chennai", "Delhi-Jaipur", "Mumbai-Delhi"}
    assert parsed.ambiguous is True


def test_no_route_mentioned_is_unmatched(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("why are freight costs rising in general", routes)
    assert parsed.routes == []


def test_year_and_month_both_parsed(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("Mumbai-Pune costs in September 2025", routes)
    assert parsed.routes == ["Mumbai-Pune"]
    assert parsed.month == 9
    assert parsed.year == 2025


# ---------------------------------------------------------------------------
# Route matching: "X to Y" / "X and Y" phrasing, not just the exact "X-Y" spelling
# ---------------------------------------------------------------------------


def test_route_named_matches_hyphen_spelling():
    assert _route_named("what happened on delhi-chennai", "Delhi", "Chennai") is True


def test_route_named_matches_to_phrasing():
    assert _route_named("why did delhi to chennai get pricier", "Delhi", "Chennai") is True


def test_route_named_matches_reversed_order():
    assert _route_named("compare chennai and delhi costs", "Delhi", "Chennai") is True


def test_route_named_matches_from_x_to_y_phrasing():
    assert _route_named("shipments from chennai to delhi last month", "Delhi", "Chennai") is True


def test_route_named_false_when_cities_are_far_apart_in_the_sentence():
    far_apart = "delhi has had a rough quarter overall, meanwhile completely separately chennai " \
                "also saw some changes"
    assert _route_named(far_apart, "Delhi", "Chennai") is False


def test_route_named_false_when_only_one_city_present():
    assert _route_named("what happened on the delhi route", "Delhi", "Chennai") is False


def test_delhi_to_chennai_phrasing_resolves_to_the_one_route_not_the_ambiguous_fallback(weekly_metrics_df):
    """Regression: this used to fall through to the city-union fallback and return every route
    touching either city (Delhi-Chennai, Delhi-Jaipur, Mumbai-Delhi, Chennai-Bangalore) instead of
    resolving to the one route actually meant."""
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("why did Delhi to Chennai get pricier", routes)
    assert parsed.routes == ["Delhi-Chennai"]
    assert parsed.ambiguous is False


def test_from_x_to_y_full_sentence_resolves_cleanly(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("what's the situation from Chennai to Delhi this quarter", routes)
    assert parsed.routes == ["Delhi-Chennai"]
    assert parsed.ambiguous is False


def test_multiple_routes_named_with_to_phrasing_are_all_matched_not_ambiguous(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("compare Delhi to Chennai and Mumbai to Pune", routes)
    assert set(parsed.routes) == {"Delhi-Chennai", "Mumbai-Pune"}
    assert parsed.ambiguous is False


# ---------------------------------------------------------------------------
# Month parsing: multiple months named in one question
# ---------------------------------------------------------------------------


def test_single_month_is_not_flagged_ambiguous(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("Mumbai-Pune costs in September 2025", routes)
    assert parsed.months_mentioned == [9]
    assert parsed.ambiguous_month is False


def test_two_months_named_are_both_captured_and_flagged_ambiguous(weekly_metrics_df):
    """Regression: previously the parser took the first MONTHS-dict match and silently dropped
    the second month with no indication anything was ignored."""
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("was Chennai-Bangalore pricier in February or March 2025", routes)
    assert parsed.months_mentioned == [2, 3]
    assert parsed.ambiguous_month is True
    assert parsed.month is None  # no single month silently chosen
    assert parsed.year == 2025


def test_two_months_named_regardless_of_which_is_mentioned_first(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("was it March or February that was worse", routes)
    assert parsed.months_mentioned == [2, 3]  # sorted, not sentence order
    assert parsed.ambiguous_month is True


def test_no_month_named_is_not_ambiguous(weekly_metrics_df):
    routes = known_routes(weekly_metrics_df)
    parsed = parse_question("how is Mumbai-Pune doing overall", routes)
    assert parsed.months_mentioned == []
    assert parsed.ambiguous_month is False
    assert parsed.month is None


def test_build_context_no_route_returns_no_data_reason(weekly_metrics_df, output_df, notes_df):
    from watchdog.ask import ParsedQuery

    parsed = ParsedQuery(routes=[], year=None, month=None, ambiguous=False)
    ctx = build_context(parsed, weekly_metrics_df, output_df, notes_df)
    assert ctx.no_data_reason == "no_route"


def test_build_context_window_with_no_data_is_flagged(weekly_metrics_df, output_df, notes_df):
    """The dataset only runs 2024-01 to 2025-12 (see test_weekly.py) -- 2030 has no rows for any
    route, so this must report no data rather than silently falling back to unfiltered rows."""
    from watchdog.ask import ParsedQuery

    parsed = ParsedQuery(routes=["Chennai-Bangalore"], year=2030, month=None, ambiguous=False)
    ctx = build_context(parsed, weekly_metrics_df, output_df, notes_df)
    assert ctx.no_data_reason == "no_window_data"


def test_build_context_matched_window_only_includes_that_routes_rows(weekly_metrics_df, output_df, notes_df):
    from watchdog.ask import ParsedQuery

    parsed = ParsedQuery(routes=["Chennai-Bangalore"], year=2025, month=3, ambiguous=False)
    ctx = build_context(parsed, weekly_metrics_df, output_df, notes_df)
    assert ctx.no_data_reason is None
    assert ctx.windowed_rows, "expected at least one Chennai-Bangalore week in March 2025"
    assert all(r["route"] == "Chennai-Bangalore" for r in ctx.windowed_rows)
    for r in ctx.windowed_rows:
        assert r["week_of"].startswith("2025-03")


def test_build_context_candidate_rows_carry_the_gates_own_reason(weekly_metrics_df, output_df, notes_df):
    """Chennai-Bangalore 2025-02-24 is justified by N001 in output.csv -- the context handed to the
    LLM must carry that exact reason/note id, not a re-derived one, so the LLM can't drift from
    what the gate actually decided."""
    from watchdog.ask import ParsedQuery

    parsed = ParsedQuery(routes=["Chennai-Bangalore"], year=None, month=None, ambiguous=False)
    ctx = build_context(parsed, weekly_metrics_df, output_df, notes_df)
    matches = [r for r in ctx.candidate_rows if r["week_of"] == "2025-02-24"]
    assert len(matches) == 1
    assert matches[0]["matched_note_id"] == "N001"
    assert "N001" in ctx.note_texts


def test_verdict_line_matches_actual_gate_outcomes(output_df):
    """Regression test for a real bug found in live testing: the LLM once said 'Delhi-Jaipur: 3
    flagged weeks, all unexplained' when one of those three is actually justified by N003. This
    deterministic ground-truth line is what lets that kind of slip be caught by inspection instead
    of trusted."""
    from watchdog.ask import _verdict_line

    rows = output_df[output_df["route"] == "Delhi-Jaipur"].to_dict("records")
    line = _verdict_line("Delhi-Jaipur", rows)
    assert "3 flagged week(s)" in line
    assert "1 justified (N003)" in line
    assert "2 unexplained" in line


def test_verdict_line_for_route_with_no_candidates(output_df):
    from watchdog.ask import _verdict_line

    line = _verdict_line("Mumbai-Pune", [])
    assert line == "Mumbai-Pune: no flagged/candidate weeks in this period."


def test_build_context_route_never_flagged_says_so(weekly_metrics_df, output_df, notes_df):
    """A route/period with zero candidates must come through with an empty candidate_rows list --
    that's what lets answer_question honestly say 'nothing was flagged' instead of the LLM
    inventing a reason to fill the gap."""
    from watchdog.ask import ParsedQuery

    parsed = ParsedQuery(routes=["Delhi-Jaipur"], year=2024, month=1, ambiguous=False)
    ctx = build_context(parsed, weekly_metrics_df, output_df, notes_df)
    assert ctx.no_data_reason is None
    assert ctx.candidate_rows == []

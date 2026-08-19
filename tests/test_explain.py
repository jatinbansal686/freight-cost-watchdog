from watchdog.explain import ReasonContext, _validate_llm_reason
from watchdog.notes.enrich import EnrichedNote


def make_note(note_id="N002"):
    return EnrichedNote(
        note_id=note_id,
        date="2025-01-20",
        applies_to="Ahmedabad-Mumbai",
        raw_text="a temporary surcharge applied by transporters",
        effective_from="2025-01-20",
        effective_to="2025-01-20",
        indicates_cost_increase=True,
        applies_to_dataset=True,
        evidence_span="a temporary surcharge applied by transporters",
    )


def make_ctx(**overrides):
    defaults = dict(
        route="Ahmedabad-Mumbai",
        week_of="2025-01-20",
        cost_per_tonne_km=3.29,
        vs_own_history_pct=0.295,
        vs_similar_routes_pct=0.225,
        verdict="justified",
        matched_note=make_note(),
        near_miss_note=None,
        near_miss_check=None,
    )
    defaults.update(overrides)
    return ReasonContext(**defaults)


def test_mentioning_the_required_note_id_does_not_trip_number_validation():
    """Regression: the digits inside 'N002' itself must not be treated as an invented number just
    because the reason was required to state the note id."""
    ctx = make_ctx()
    text = (
        "Costs are justified: N002 dated 2025-01-20 describes a temporary surcharge applied by "
        "transporters, matching the +29.5% rise vs history and +22.5% vs similar routes."
    )
    assert _validate_llm_reason(text, ctx) is True


def test_an_extra_note_id_is_still_rejected():
    ctx = make_ctx()
    text = "N002 and also N005 both explain this rise of +29.5% vs history and +22.5% vs similar routes."
    assert _validate_llm_reason(text, ctx) is False


def test_missing_the_required_note_id_is_rejected():
    ctx = make_ctx()
    text = "This is justified by a temporary surcharge, a +29.5% rise vs history and +22.5% vs similar routes."
    assert _validate_llm_reason(text, ctx) is False


def test_an_invented_number_is_rejected():
    ctx = make_ctx()
    text = "N002 dated 2025-01-20 explains a +50.0% rise vs history and +22.5% vs similar routes."
    assert _validate_llm_reason(text, ctx) is False

import inspect

from watchdog.explain import ReasonContext, build_reason, _validate_llm_reason
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


def test_build_reason_max_tokens_has_no_default():
    """max_tokens must come from config.yaml (explain.max_tokens), not a hardcoded fallback here --
    a bare default on this signature would be a second, un-synced source of truth for the same
    number config.yaml already owns."""
    sig = inspect.signature(build_reason)
    assert sig.parameters["max_tokens"].default is inspect.Parameter.empty


def test_build_reason_template_mode_needs_no_client_or_llm_call():
    ctx = make_ctx()
    text = build_reason(ctx, mode="template", client=None, cache={}, max_tokens=123)
    assert "N002" in text
    assert "29.5" in text and "22.5" in text


def test_build_reason_uses_cache_before_calling_the_client():
    ctx = make_ctx()
    key_holder = {}

    class _ExplodingClient:
        available = True

        def complete(self, *a, **kw):
            raise AssertionError("should not be called when the cache already has this key")

    from watchdog.explain import _cache_key

    cache = {_cache_key(ctx): "a cached reason string"}
    text = build_reason(ctx, mode="llm", client=_ExplodingClient(), cache=cache, max_tokens=123)
    assert text == "a cached reason string"

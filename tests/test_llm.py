"""Tests for llm.py: the truncation-recovery heuristic (_looks_complete), LLMClient's
retry/truncation/error handling, and CostTracker's accounting. No real network calls --
LLMClient._client is swapped for an in-process fake that returns pre-scripted responses, so these
run offline and deterministically.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from watchdog.config import LLMConfig
from watchdog.llm import CostTracker, LLMClient, _looks_complete

# ---------------------------------------------------------------------------
# _looks_complete -- the truncation-recovery heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected,reason",
    [
        ("", False, "empty string is never complete"),
        ("   ", False, "whitespace-only is never complete"),
        ("the cost rose because", False, "mid-word cutoff, no terminal punctuation"),
        ("The cost rose due to diesel prices.", True, "ordinary sentence"),
        ("Was it justified?", True, "question mark"),
        ("He said it was fine!", True, "exclamation mark"),
        (
            "...with a 29.5 % increase over own history and a 22.",
            False,
            "truncated decimal -- '22.' cut off before '5%', digit before the period",
        ),
        (
            "...supported by note N003 dated 2025-05-05.",
            True,
            "genuine sentence ending in an ISO date -- must not be rejected as a fake decimal",
        ),
        (
            'The note said, "as noted in N003 on 2025-05-05."',
            True,
            "ISO date ending closed by a balanced double quote",
        ),
        (
            'The reason given was "temporary surcharge."',
            True,
            "balanced double quotes closing a real sentence",
        ),
        (
            '...supported by evidence, the driver said "',
            False,
            "dangling *open* double quote -- odd quote count, unmatched",
        ),
        ("...it worked.'", True, "apostrophe closing real sentence-ending punctuation"),
        (
            "...supported by evidence, the driver said '",
            False,
            "dangling open single quote with no preceding terminal punctuation",
        ),
        ("...the route'", False, "truncated mid-possessive -- apostrophe not closing anything"),
        ("the routes' costs rose.", True, "apostrophe mid-sentence is irrelevant; real ending is '.'"),
        ("2025", False, "bare year, no terminal punctuation at all"),
        ("...on 05-05.", False, "two-digit-only tail is not a full ISO date, still looks like a truncated decimal"),
    ],
)
def test_looks_complete(text, expected, reason):
    assert _looks_complete(text) is expected, reason


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def test_cost_tracker_aggregates_across_purposes():
    tracker = CostTracker()
    tracker.record("note_enrichment", SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    tracker.record("note_enrichment", SimpleNamespace(prompt_tokens=50, completion_tokens=10))
    tracker.record("explanation", SimpleNamespace(prompt_tokens=30, completion_tokens=5))

    assert tracker.total_calls == 3
    assert tracker.total_prompt_tokens == 180
    assert tracker.total_completion_tokens == 35
    assert tracker.by_purpose() == {"note_enrichment": 2, "explanation": 1}


def test_cost_tracker_handles_missing_usage_as_zero():
    """A response with no usage object (e.g. a provider quirk) must not crash accounting -- it
    should just record zero tokens for that call, not raise or silently skip the call count."""
    tracker = CostTracker()
    tracker.record("qa", None)
    assert tracker.total_calls == 1
    assert tracker.total_prompt_tokens == 0
    assert tracker.total_completion_tokens == 0


# ---------------------------------------------------------------------------
# LLMClient -- fake OpenAI-compatible client, no network
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content, finish_reason, prompt_tokens=10, completion_tokens=5, usage=True):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)]
        self.usage = (
            SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens) if usage else None
        )


class _FakeCompletions:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("LLMClient made more calls than the test scripted")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAI:
    def __init__(self, scripted):
        self.completions = _FakeCompletions(scripted)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_client(scripted, **cfg_overrides):
    cfg = LLMConfig(
        base_url="http://fake.invalid",
        model="fake-model",
        temperature=0,
        top_p=1,
        max_tokens=100,
        max_retries=4,
        retry_backoff_seconds=0,
        retry_max_tokens_cap=800,
        api_key="fake-key",
    )
    cfg = replace(cfg, **cfg_overrides)
    tracker = CostTracker()
    client = LLMClient(cfg, tracker=tracker)
    client._client = _FakeOpenAI(scripted)
    return client, tracker


def test_available_is_false_without_api_key():
    cfg = LLMConfig(
        base_url="http://fake.invalid", model="m", temperature=0, top_p=1, max_tokens=100,
        max_retries=1, retry_backoff_seconds=0, retry_max_tokens_cap=800, api_key=None,
    )
    client = LLMClient(cfg)
    assert client.available is False
    assert client.complete("sys", "user", purpose="qa") is None


def test_complete_returns_content_on_clean_stop():
    client, tracker = _make_client([_FakeResponse("a full sentence.", finish_reason="stop")])
    result = client.complete("sys", "user", purpose="qa")
    assert result == "a full sentence."
    assert tracker.total_calls == 1
    assert tracker.by_purpose() == {"qa": 1}


def test_complete_accepts_truncated_response_that_looks_complete():
    """finish_reason == 'length' but the content already ends at a real sentence boundary --
    should be returned as-is, no retry."""
    client, tracker = _make_client([_FakeResponse("done at a boundary.", finish_reason="length")])
    result = client.complete("sys", "user", purpose="qa")
    assert result == "done at a boundary."
    assert len(client._client.completions.calls) == 1


def test_complete_retries_a_genuinely_truncated_response():
    scripted = [
        _FakeResponse("...cut off mid decimal 22.", finish_reason="length"),
        _FakeResponse("...cut off mid decimal 22.5%, now complete.", finish_reason="stop"),
    ]
    client, tracker = _make_client(scripted, max_tokens=50)
    result = client.complete("sys", "user", purpose="explanation")
    assert result == "...cut off mid decimal 22.5%, now complete."
    assert len(client._client.completions.calls) == 2
    # second attempt must have doubled the budget
    assert client._client.completions.calls[0]["max_tokens"] == 50
    assert client._client.completions.calls[1]["max_tokens"] == 100
    assert tracker.total_calls == 2  # both attempts recorded, even the discarded truncated one


def test_complete_retry_budget_respects_the_configured_cap():
    scripted = [
        _FakeResponse("cut off 1.", finish_reason="length"),
        _FakeResponse("cut off 2.", finish_reason="length"),
        _FakeResponse("finally complete.", finish_reason="stop"),
    ]
    client, _ = _make_client(scripted, max_tokens=500, retry_max_tokens_cap=800, max_retries=5)
    result = client.complete("sys", "user", purpose="explanation")
    assert result == "finally complete."
    budgets = [c["max_tokens"] for c in client._client.completions.calls]
    assert budgets == [500, 800, 800]  # 500*2=1000 clamped to the 800 cap, then stays at the cap


def test_complete_returns_none_after_exhausting_all_retries_on_truncation():
    scripted = [_FakeResponse(f"still cut off {i}.", finish_reason="length") for i in range(3)]
    client, _ = _make_client(scripted, max_retries=3, max_tokens=50)
    result = client.complete("sys", "user", purpose="qa")
    assert result is None
    assert len(client._client.completions.calls) == 3


def test_complete_retries_on_transient_exception_then_succeeds():
    scripted = [RuntimeError("rate limited"), _FakeResponse("recovered fine.", finish_reason="stop")]
    client, tracker = _make_client(scripted, max_retries=3)
    result = client.complete("sys", "user", purpose="qa")
    assert result == "recovered fine."
    # the failed attempt never reached response.usage, so only the successful call is tracked
    assert tracker.total_calls == 1


def test_complete_returns_none_when_every_attempt_raises():
    scripted = [RuntimeError("down"), RuntimeError("still down"), RuntimeError("still down")]
    client, _ = _make_client(scripted, max_retries=3)
    result = client.complete("sys", "user", purpose="qa")
    assert result is None


def test_complete_passes_explicit_max_tokens_override_not_config_default():
    client, _ = _make_client([_FakeResponse("ok.", finish_reason="stop")], max_tokens=100)
    client.complete("sys", "user", purpose="qa", max_tokens=333)
    assert client._client.completions.calls[0]["max_tokens"] == 333

"""Thin wrapper around the NVIDIA NIM (OpenAI-compatible) endpoint.

Every call goes through here so call/token counts are tracked in one place for the cost log
(outputs/token_cost_log.md). temperature is always read from config and is 0 -- reproducibility
is a graded guardrail, never raised regardless of caller.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from watchdog.config import LLMConfig


@dataclass
class CallRecord:
    purpose: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CostTracker:
    calls: list[CallRecord] = field(default_factory=list)

    def record(self, purpose: str, usage) -> None:
        if usage is None:
            self.calls.append(CallRecord(purpose, 0, 0))
            return
        self.calls.append(CallRecord(purpose, usage.prompt_tokens, usage.completion_tokens))

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def by_purpose(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.purpose] = out.get(c.purpose, 0) + 1
        return out


_ISO_DATE_TAIL_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.$")


def _looks_complete(content: str) -> bool:
    """True if truncated content still ends at a real sentence boundary.

    Three lexical traps this guards against:
    - A truncated decimal (e.g. "...a 22." cut off before "5%") ends in digit + period, which
      looks like a sentence end but isn't -- rejected, unless the digits form a full ISO date
      ("...on 2025-05-05.", a genuine sentence ending that shows up throughout this dataset's
      reasons and would otherwise trigger a needless retry).
    - A truncation landing mid-double-quote leaves a dangling *open* quote as the last
      character. `"` is normally a fine sentence-ending character (a quoted clause closing),
      but only if it's actually closing something -- an odd total count of `"` means the last
      one is unmatched.
    - A trailing `'` is ambiguous on its own: it's just as likely to be a truncated possessive
      ("...the route'" cut off before "s") as a genuine closing single-quote. Unlike `"`, a
      running parity count doesn't help here -- contractions/possessives make an odd apostrophe
      count normal in ordinary prose. Instead, only accept a trailing `'` when it's closing real
      sentence-ending punctuation ("...it worked.'"); anything else is treated as incomplete and
      retried, since a wasted retry is a cheap price for never silently keeping cut-off text.
    """
    stripped = content.strip()
    if not stripped:
        return False
    if stripped.count('"') % 2 == 1:
        return False
    last = stripped[-1]
    if last == "'":
        return len(stripped) >= 2 and stripped[-2] in ".!?"
    if last not in ".!?\"":
        return False
    if last == "." and len(stripped) >= 2 and stripped[-2].isdigit() and not _ISO_DATE_TAIL_RE.search(stripped):
        return False
    return True


class LLMClient:
    def __init__(self, cfg: LLMConfig, tracker: CostTracker | None = None):
        self.cfg = cfg
        self.tracker = tracker if tracker is not None else CostTracker()
        self._client = None
        if cfg.api_key:
            self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=60.0, max_retries=0)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str, purpose: str, max_tokens: int | None = None) -> str | None:
        """Returns the completion text, or None if the client has no key or every retry failed.

        This model emits internal `reasoning_content` before its final `content`, spending part of
        `max_tokens` on it. If the response gets cut off mid-reasoning (`finish_reason == "length"`
        with no usable content, or a clearly unterminated sentence), we retry once with a larger
        budget rather than silently returning a truncated explanation.
        """
        if not self.available:
            return None

        budget = max_tokens or self.cfg.max_tokens
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=budget,
                )
                self.tracker.record(purpose, resp.usage)
                content = resp.choices[0].message.content
                truncated = resp.choices[0].finish_reason == "length"
                if content and not truncated:
                    return content.strip()
                if content and truncated and _looks_complete(content):
                    return content.strip()  # cut off right at a sentence boundary -- usable
                budget = min(budget * 2, self.cfg.retry_max_tokens_cap)
                last_error = RuntimeError(f"truncated response (finish_reason=length), retrying with max_tokens={budget}")
            except Exception as exc:  # noqa: BLE001 -- NIM free tier can raise rate-limit/transient errors
                last_error = exc
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(self.cfg.retry_backoff_seconds * (2**attempt))
        print(f"[llm] '{purpose}' failed after {self.cfg.max_retries} attempts: {last_error}")
        return None

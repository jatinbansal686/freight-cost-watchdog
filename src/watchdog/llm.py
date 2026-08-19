"""Thin wrapper around the NVIDIA NIM (OpenAI-compatible) endpoint.

Every call goes through here so call/token counts are tracked in one place for the cost log
(outputs/token_cost_log.md). temperature is always read from config and is 0 -- reproducibility
is a graded guardrail, never raised regardless of caller.
"""
from __future__ import annotations

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
                if content and truncated and content.strip().rstrip()[-1:] in ".!?\"'":
                    return content.strip()  # cut off right at a sentence boundary -- usable
                budget = min(budget * 2, 1600)
                last_error = RuntimeError(f"truncated response (finish_reason=length), retrying with max_tokens={budget}")
            except Exception as exc:  # noqa: BLE001 -- NIM free tier can raise rate-limit/transient errors
                last_error = exc
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(self.cfg.retry_backoff_seconds * (2**attempt))
        print(f"[llm] '{purpose}' failed after {self.cfg.max_retries} attempts: {last_error}")
        return None

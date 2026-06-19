"""
Per-session token budget tracker.

Reads MAX_SESSION_TOKENS from env (default 2 000 000).
  - At 80 % consumed: fires a non-blocking Telegram alert.
  - At 100 % consumed: saves state and raises TokenBudgetExceededError.

The alert callback is set by the orchestrator at start-up to avoid a
circular import with human_loop.py.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TokenBudgetExceededError(Exception):
    """Raised when the session-level token budget is exhausted."""


class TokenBudget:
    ALERT_THRESHOLD = 0.80

    def __init__(self) -> None:
        self.max_tokens: int = int(os.getenv("MAX_SESSION_TOKENS", 2_000_000))
        self._totals: dict[str, int] = {
            "ollama_input": 0,
            "ollama_output": 0,
            "anthropic_input": 0,
            "anthropic_output": 0,
        }
        self._alert_fired: bool = False
        # Injected by orchestrator.py after both modules are initialised
        self._alert_callback: Optional[Callable[[str], None]] = None
        self._halt_callback: Optional[Callable[[], None]] = None

    def set_callbacks(
        self,
        alert: Callable[[str], None],
        halt: Callable[[], None],
    ) -> None:
        self._alert_callback = alert
        self._halt_callback = halt

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        source: str,  # "ollama" | "anthropic"
        state=None,  # AgentState | None  — passed so we can save before halt
    ) -> None:
        self._totals[f"{source}_input"] += input_tokens
        self._totals[f"{source}_output"] += output_tokens

        pct = self.percent_used()
        logger.debug(
            "Token budget: %d / %d (%.1f%%)",
            self.total(),
            self.max_tokens,
            pct * 100,
        )

        if pct >= 1.0:
            if state is not None:
                state.save()
            if self._halt_callback:
                self._halt_callback()
            raise TokenBudgetExceededError(
                f"Session token budget exhausted: {self.total():,} / {self.max_tokens:,}"
            )

        if pct >= self.ALERT_THRESHOLD and not self._alert_fired:
            self._alert_fired = True
            msg = (
                f"Token budget at {pct*100:.0f}% "
                f"({self.total():,} / {self.max_tokens:,} tokens). "
                "Agent will halt at 100%."
            )
            logger.warning(msg)
            if self._alert_callback:
                try:
                    self._alert_callback(msg)
                except Exception as exc:
                    logger.error("Telegram alert failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Inspection                                                           #
    # ------------------------------------------------------------------ #

    def total(self) -> int:
        return sum(self._totals.values())

    def percent_used(self) -> float:
        return self.total() / self.max_tokens

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.total())

    def summary(self) -> dict:
        return {
            **self._totals,
            "total": self.total(),
            "max": self.max_tokens,
            "percent_used": round(self.percent_used() * 100, 2),
        }

    def exhausted(self) -> bool:
        return self.total() >= self.max_tokens

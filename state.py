"""
AgentState: single source of truth for the RL loop.
Serialised atomically to state.json after every iteration and before every
Telegram escalation so the process can be killed and resumed at any point.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentState:
    # RL loop position
    phase: str = "A"                      # "A" | "C" | "done" | "failed"
    iteration: int = 0                    # total iterations run so far
    consecutive_plus3: int = 0           # Phase B gate counter (resets on non-+3)
    gate_passed: bool = False            # True once Phase B unlocks Phase C
    batch_size_index: int = 0            # index into PHASE_B_BATCH_SIZES

    # Best result tracking
    best_reward: int = -999
    best_kernel_path: str = ""

    # Conversation context (full LLM message log)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    # Token accounting
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "ollama_input": 0,
        "ollama_output": 0,
        "anthropic_input": 0,
        "anthropic_output": 0,
    })

    # Timestamps
    start_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Phase C outcomes (populated on success)
    phase_c_results: dict[str, float] = field(default_factory=dict)

    # Audit counters
    human_interventions: int = 0
    last_compile_result: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str = "state.json") -> None:
        """Atomic write via temp-file rename to prevent corruption on kill."""
        data = asdict(self)
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str = "state.json") -> "AgentState":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def new(cls) -> "AgentState":
        return cls()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def elapsed_hours(self) -> float:
        start = datetime.fromisoformat(self.start_time)
        now = datetime.now(timezone.utc)
        return (now - start).total_seconds() / 3600.0

    def total_tokens(self) -> int:
        return sum(self.token_usage.values())

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        source: str,  # "ollama" | "anthropic"
    ) -> None:
        self.token_usage[f"{source}_input"] += input_tokens
        self.token_usage[f"{source}_output"] += output_tokens

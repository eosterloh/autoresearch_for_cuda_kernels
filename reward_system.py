"""
Reward function and Phase B gate for the RL loop.

Reward signal
-------------
  -1  Compile failure OR correctness check failed
  +1  Correct output but slower than torch.compile baseline
  +3  Correct output AND faster than torch.compile baseline

Phase B gate
------------
The agent must earn +3 across 3 consecutive evaluations, each at a different
batch size drawn from PHASE_B_BATCH_SIZES = [1, 32, 128].

The gate cycles through batch sizes sequentially. Any non-+3 reward resets
both the counter AND the batch size index back to the beginning, ensuring the
agent demonstrates stable performance across the full range before advancing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import AgentState

logger = logging.getLogger(__name__)

PHASE_B_BATCH_SIZES: list[int] = [1, 32, 128]
REQUIRED_CONSECUTIVE: int = len(PHASE_B_BATCH_SIZES)


# ------------------------------------------------------------------ #
# Reward evaluation                                                    #
# ------------------------------------------------------------------ #

def evaluate(
    compile_result: dict,
    speed_result: dict,
    benchmark_result: dict,
) -> int:
    """
    Compute the reward signal from tool outputs.

    Args
    ----
    compile_result   : output of SyntaxCompileTool
    speed_result     : output of SandboxSpeedTestTool
    benchmark_result : output of BenchmarkTool

    Returns
    -------
    -1, +1, or +3
    """
    if not compile_result.get("success", False):
        logger.info("Reward -1: compilation failed")
        return -1

    if not speed_result.get("correctness_passed", False):
        logger.info("Reward -1: correctness check failed (torch.allclose)")
        return -1

    speedup = benchmark_result.get("speedup_ratio", 0.0)
    if speedup >= 1.0:
        logger.info("Reward +3: correct and faster than torch.compile (speedup=%.3f)", speedup)
        return 3

    logger.info("Reward +1: correct but not faster than torch.compile (speedup=%.3f)", speedup)
    return 1


def reward_explanation(
    reward: int,
    speed_result: dict,
    benchmark_result: dict,
) -> str:
    """Human-readable explanation injected into the conversation history."""
    if reward == -1:
        if not speed_result.get("correctness_passed", True):
            return (
                "PENALTY: Kernel output failed torch.allclose(atol=1e-2, rtol=1e-2) "
                "against FP16 reference. Fix the dequantization math."
            )
        return (
            "PENALTY: Kernel failed to compile. "
            "Fix all syntax and type errors shown in compiler_traceback."
        )
    if reward == 1:
        speedup = benchmark_result.get("speedup_ratio", 0.0)
        tc_ms = benchmark_result.get("torch_compile_latency_ms", 0.0)
        k_ms = benchmark_result.get("kernel_latency_ms", 0.0)
        return (
            f"BASELINE: Kernel is mathematically correct but slower than torch.compile. "
            f"Your latency: {k_ms:.3f}ms, torch.compile: {tc_ms:.3f}ms, speedup: {speedup:.3f}x. "
            "Optimise memory access patterns, shared memory usage, or thread block geometry."
        )
    # reward == 3
    speedup = benchmark_result.get("speedup_ratio", 0.0)
    k_ms = benchmark_result.get("kernel_latency_ms", 0.0)
    return (
        f"OPTIMISED: Kernel is correct and faster than torch.compile. "
        f"Latency: {k_ms:.3f}ms, speedup: {speedup:.3f}x. "
        "Gate progress advances. Continue to next batch size."
    )


# ------------------------------------------------------------------ #
# Phase B gate                                                         #
# ------------------------------------------------------------------ #

class PhaseBGate:
    """
    Tracks consecutive +3 rewards across PHASE_B_BATCH_SIZES.
    Updates and checks are applied directly to AgentState.
    """

    def current_batch_size(self, state: "AgentState") -> int:
        idx = min(state.batch_size_index, len(PHASE_B_BATCH_SIZES) - 1)
        return PHASE_B_BATCH_SIZES[idx]

    def update(self, reward: int, state: "AgentState") -> None:
        if reward == 3:
            state.consecutive_plus3 += 1
            state.batch_size_index = min(
                state.batch_size_index + 1, len(PHASE_B_BATCH_SIZES)
            )
            logger.info(
                "Phase B gate: %d / %d consecutive +3 rewards",
                state.consecutive_plus3,
                REQUIRED_CONSECUTIVE,
            )
        else:
            if state.consecutive_plus3 > 0:
                logger.info(
                    "Phase B gate reset (was at %d / %d)",
                    state.consecutive_plus3,
                    REQUIRED_CONSECUTIVE,
                )
            state.consecutive_plus3 = 0
            state.batch_size_index = 0

    def check(self, state: "AgentState") -> bool:
        return state.consecutive_plus3 >= REQUIRED_CONSECUTIVE


gate = PhaseBGate()

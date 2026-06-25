"""
Main orchestrator: the autonomous RL loop that drives the agent.

Start fresh:    python orchestrator.py
Resume run:     python orchestrator.py --resume

Loop lifecycle
--------------
Phase A  — sandbox micro-loop: draft → compile → speed test → reward
Phase B  — gate: 3 consecutive +3 rewards across batch sizes [1, 32, 128]
Phase C  — production: load Gemma 3 12B AWQ, monkey-patch, benchmark TTFT/TPS
           On Phase C failure: feed traceback back into Phase A and continue

Hard stops
----------
  • 24-hour wall-clock budget (TIME_BUDGET_HOURS env var)
  • Per-session token budget (MAX_SESSION_TOKENS env var)
  • Human escalation via Telegram on irrecoverable errors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import llm_client
import reward_system
import tools as tool_module
from human_loop import human_loop
from state import AgentState
from token_budget import TokenBudget, TokenBudgetExceededError

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()
STATE_FILE = str(PROJECT_ROOT / "state.json")

TIME_BUDGET_HOURS = float(os.getenv("TIME_BUDGET_HOURS", "24"))
CONTEXT_WINDOW_MAX_MESSAGES = int(os.getenv("CONTEXT_WINDOW_MAX_MESSAGES", "40"))

# ------------------------------------------------------------------ #
# System prompt                                                        #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an autonomous CUDA kernel optimisation agent running on a DGX Spark
with an NVIDIA GB10 Grace Blackwell GPU (compute capability sm_100).

YOUR MISSION
------------
Generate, compile, and iteratively optimise a custom CUDA C++ kernel that performs
INT4 → FP16 fused dequantization matrix multiplication (AWQ format, group size 128).

The kernel must outperform torch.compile on the same operation on this hardware.

REWARD SIGNAL
-------------
After each SandboxSpeedTestTool + BenchmarkTool evaluation, you will receive one of:
  REWARD: -1  — Compile failure OR incorrect output (fix before anything else)
  REWARD: +1  — Correct but slower than torch.compile (optimise further)
  REWARD: +3  — Correct AND faster than torch.compile (advance toward Phase B gate)

PHASE B GATE
------------
You must achieve REWARD: +3 for three consecutive batch sizes [1, 32, 128] (in order).
Any non-+3 result resets the counter. Once the gate passes, Phase C begins automatically.

TOOL SCHEMAS
------------
""" + json.dumps(tool_module.TOOL_SCHEMAS, indent=2) + """

RESPONSE FORMAT
---------------
You MUST respond with EXACTLY one JSON object and nothing else:
{
  "tool_call": {
    "name": "<ToolName>",
    "arguments": { ... }
  }
}

No markdown. No prose. No explanation. JSON only.

STRATEGY GUIDANCE
-----------------
1. Start by querying RAGKernelCorpusTool for INT4 AWQ dequantization examples.
2. Write current_kernel.cu with the dequantize_matmul __global__ kernel.
3. Write binding.cpp with the pybind11 wrapper exposing dequantize_matmul().
4. Compile with SyntaxCompileTool. Fix all errors before proceeding.
5. Test with SandboxSpeedTestTool at batch_size=1 first.
6. Benchmark with BenchmarkTool. Iterate on shared memory, thread block geometry,
   vectorised loads (__ldg), and warp-level primitives (__shfl_*).
7. Target: fused INT4 unpack + scale/zero-point application + GEMM in a single pass.

AWQ INT4 WEIGHT LAYOUT
-----------------------
Weights are packed: 8 INT4 values per int32, lower nibble first.
Group size: 128 (one scale + zero-point per 128 weights).
Dequant formula: weight_fp16 = (int4_val - zero_point) * scale
Output dtype: FP16 (torch.float16).
"""


# ------------------------------------------------------------------ #
# Context window management                                            #
# ------------------------------------------------------------------ #

def _trim_history(state: AgentState) -> None:
    """
    If conversation_history exceeds CONTEXT_WINDOW_MAX_MESSAGES,
    summarise the oldest half and replace with a single summary message.
    """
    if len(state.conversation_history) <= CONTEXT_WINDOW_MAX_MESSAGES:
        return

    n_old = len(state.conversation_history) - CONTEXT_WINDOW_MAX_MESSAGES // 2
    old_messages = state.conversation_history[:n_old]
    recent_messages = state.conversation_history[n_old:]

    logger.info(
        "Context window at %d messages — summarising %d old messages",
        len(state.conversation_history),
        n_old,
    )
    summary_text = llm_client.summarise_history(old_messages)
    summary_message = {
        "role": "user",
        "content": f"[HISTORY SUMMARY — {n_old} earlier messages condensed]\n\n{summary_text}",
    }
    state.conversation_history = [summary_message] + recent_messages


# ------------------------------------------------------------------ #
# Reward injection                                                      #
# ------------------------------------------------------------------ #

def _build_messages(state: AgentState) -> list[dict]:
    """Prepend the system prompt to the conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + state.conversation_history


def _append_tool_result(state: AgentState, tool_name: str, result: dict) -> None:
    state.conversation_history.append({
        "role": "user",
        "content": (
            f"[TOOL RESULT: {tool_name}]\n"
            + json.dumps(result, indent=2, default=str)
        ),
    })


def _append_assistant(state: AgentState, tool_call: dict) -> None:
    state.conversation_history.append({
        "role": "assistant",
        "content": json.dumps({"tool_call": tool_call}),
    })


# ------------------------------------------------------------------ #
# Phase A — sandbox micro-loop                                         #
# ------------------------------------------------------------------ #

def _run_phase_a_step(state: AgentState, budget: TokenBudget) -> None:
    """Execute one iteration of the Phase A loop."""
    _trim_history(state)

    messages = _build_messages(state)
    tool_call = llm_client.call(messages)

    tool_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})

    logger.info("Iteration %d | Tool: %s | Args: %s", state.iteration, tool_name, arguments)
    _append_assistant(state, tool_call)

    result = tool_module.dispatch(tool_name, arguments)
    _append_tool_result(state, tool_name, result)

    # Store compile result for reward evaluation later
    if tool_name == "SyntaxCompileTool":
        state.last_compile_result = result

    # Evaluate reward when a speed test completes
    if tool_name == "SandboxSpeedTestTool":
        compile_r = state.last_compile_result or {"success": False}
        bench_r = tool_module.dispatch(
            "BenchmarkTool",
            {"kernel_name": arguments.get("kernel_name", "cuda_kernel")},
        )
        _append_tool_result(state, "BenchmarkTool", bench_r)

        reward = reward_system.evaluate(compile_r, result, bench_r)
        explanation = reward_system.reward_explanation(reward, result, bench_r)

        reward_message = f"REWARD: {reward:+d}\n{explanation}"
        state.conversation_history.append({"role": "user", "content": reward_message})
        logger.info(reward_message)

        # Log this iteration
        is_best = reward > state.best_reward
        state.iteration_log.append({
            "iteration": state.iteration,
            "reward": reward,
            "speedup_ratio": bench_r.get("speedup_ratio", 0.0),
            "latency_ms": result.get("latency_ms", 0.0),
            "correctness_passed": result.get("correctness_passed", False),
            "is_best": is_best,
        })

        # Update best kernel tracking
        if is_best:
            state.best_reward = reward
            src = PROJECT_ROOT / "sandbox" / "current_kernel.cu"
            if src.exists():
                dst = PROJECT_ROOT / "sandbox" / f"best_kernel_iter{state.iteration}.cu"
                shutil.copy(src, dst)
                state.best_kernel_path = str(dst)

        # Phase B gate update
        reward_system.gate.update(reward, state)

        if reward_system.gate.check(state) and not state.gate_passed:
            state.gate_passed = True
            state.phase = "C"
            gate_msg = (
                "PHASE B GATE PASSED ✓\n"
                "You have achieved +3 across all three batch sizes [1, 32, 128].\n"
                "Advancing to Phase C: production integration with Gemma 3 12B AWQ.\n"
                "Call TransformersWeightLoaderTool to begin."
            )
            state.conversation_history.append({"role": "user", "content": gate_msg})
            logger.info("Phase B gate passed — advancing to Phase C")

    state.iteration += 1
    state.save(STATE_FILE)


# ------------------------------------------------------------------ #
# Phase C — production integration                                     #
# ------------------------------------------------------------------ #

def _run_phase_c(state: AgentState) -> bool:
    """
    Run Phase C: let the agent drive the production integration.
    Returns True if Phase C completed successfully, False if it failed
    and should revert to Phase A.
    """
    logger.info("Phase C: production integration starting")

    # Allow the agent to call TransformersWeightLoaderTool and orchestrate patching
    max_phase_c_steps = 20
    for _ in range(max_phase_c_steps):
        _trim_history(state)
        messages = _build_messages(state)

        try:
            tool_call = llm_client.call(messages)
        except RuntimeError as exc:
            human_loop.send_alert(
                "Phase C LLM failure",
                f"LLM could not produce a tool call in Phase C:\n{exc}",
                state,
            )
            reply = human_loop.wait_for_reply()
            state.conversation_history.append({"role": "user", "content": reply})
            continue

        tool_name = tool_call["name"]
        arguments = tool_call.get("arguments", {})
        _append_assistant(state, tool_call)

        result = tool_module.dispatch(tool_name, arguments)
        _append_tool_result(state, tool_name, result)
        state.iteration += 1
        state.save(STATE_FILE)

        # Check if Phase C benchmark completed
        if tool_name == "BashExecutionTool" and "ttft_ms" in result.get("stdout", ""):
            try:
                data = json.loads(result["stdout"].strip())
                state.phase_c_results = data
                state.phase = "done"
                state.save(STATE_FILE)
                logger.info("Phase C complete: TTFT=%.1fms, TPS=%.2f", data["ttft_ms"], data["tps"])
                return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Detect Phase C failure (model crash, OOM, etc.)
        if result.get("exit_code", 0) != 0 or result.get("success") is False:
            error_text = result.get("stderr") or result.get("error") or str(result)
            logger.warning("Phase C step failed: %.200s", error_text)
            revert_msg = (
                f"Phase C integration failed:\n{error_text[:1000]}\n\n"
                "Reverting to Phase A. Analyse the error and fix the kernel's "
                "memory layout or dtype handling before the next attempt."
            )
            state.conversation_history.append({"role": "user", "content": revert_msg})
            state.phase = "A"
            state.gate_passed = False
            state.consecutive_plus3 = 0
            state.batch_size_index = 0
            state.save(STATE_FILE)
            return False

    logger.warning("Phase C exceeded max steps without completing")
    return False


# ------------------------------------------------------------------ #
# Termination                                                          #
# ------------------------------------------------------------------ #

def _send_final_report(state: AgentState, reason: str, budget: TokenBudget) -> None:
    tu = state.token_usage
    ollama_total = tu["ollama_input"] + tu["ollama_output"]
    anthropic_total = tu["anthropic_input"] + tu["anthropic_output"]

    # ── Detailed stdout report ───────────────────────────────────────────
    sep = "=" * 66
    print(f"\n{sep}")
    print("  AUTORESEARCH AGENT — FINAL REPORT")
    print(sep)
    print(f"  Stop reason : {reason}")
    print(f"  Duration    : {state.elapsed_hours():.2f}h")
    print(f"  Iterations  : {state.iteration}")
    print(f"  Phase B gate: {'PASSED ✓' if state.gate_passed else 'not reached'}")
    print()
    print("  TOKEN USAGE")
    print(f"    Ollama  (local, free) : {ollama_total:>12,}  "
          f"(in {tu['ollama_input']:,} / out {tu['ollama_output']:,})")
    print(f"    Anthropic (billed)    : {anthropic_total:>12,}  "
          f"(in {tu['anthropic_input']:,} / out {tu['anthropic_output']:,})")
    print(f"    Grand total           : {state.total_tokens():>12,}")
    print(f"    Anthropic budget cap  : {budget.max_tokens:>12,}  "
          f"({budget.percent_used()*100:.1f}% used)")
    print()
    print("  BEST KERNEL")
    print(f"    Reward : {state.best_reward:+d}")
    print(f"    Path   : {state.best_kernel_path or 'none'}")

    if state.phase_c_results:
        r = state.phase_c_results
        print()
        print("  PHASE C RESULTS")
        print(f"    TTFT : {r.get('ttft_ms', 'N/A')} ms")
        print(f"    TPS  : {r.get('tps', 'N/A')} tokens/s")

    if state.iteration_log:
        print()
        print("  REWARD HISTORY  (speed-test iterations only)")
        print(f"  {'Iter':>5}  {'Reward':>7}  {'Speedup':>8}  {'Latency':>9}  {'Correct':>7}  {'Note'}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*10}")
        for e in state.iteration_log:
            note = "← BEST" if e["is_best"] else ""
            print(
                f"  {e['iteration']:>5}  "
                f"  {e['reward']:>+6}  "
                f"  {e['speedup_ratio']:>7.3f}x  "
                f"  {e['latency_ms']:>7.2f}ms  "
                f"  {'✓' if e['correctness_passed'] else '✗':>7}  "
                f"  {note}"
            )

        best_entries = [e for e in state.iteration_log if e["is_best"]]
        if best_entries:
            print()
            print("  PATH TO BEST KERNEL")
            for e in best_entries:
                print(f"    Iter {e['iteration']:>3}: reward {e['reward']:+d}, "
                      f"speedup {e['speedup_ratio']:.3f}x, "
                      f"latency {e['latency_ms']:.2f}ms")

    print(sep)
    print()

    # ── Telegram summary (brief) ─────────────────────────────────────────
    tg_lines = [
        "<b>Autoresearch Agent — Run Complete</b>",
        f"Stop reason: {reason}",
        f"Duration: {state.elapsed_hours():.2f}h  |  Iterations: {state.iteration}",
        f"Best reward: {state.best_reward:+d}  |  Gate: {'passed ✓' if state.gate_passed else 'not reached'}",
        f"Ollama tokens: {ollama_total:,} (free)",
        f"Anthropic tokens: {anthropic_total:,} / {budget.max_tokens:,} ({budget.percent_used()*100:.1f}%)",
        f"Best kernel: {state.best_kernel_path or 'none'}",
    ]
    if state.phase_c_results:
        r = state.phase_c_results
        tg_lines.append(f"Phase C — TTFT: {r.get('ttft_ms','N/A')}ms | TPS: {r.get('tps','N/A')}")

    human_loop.notify("\n".join(tg_lines))
    logger.info("Final report printed and sent")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch CUDA kernel optimisation agent")
    parser.add_argument("--resume", action="store_true", help="Resume from state.json")
    args = parser.parse_args()

    # Load or initialise state
    if args.resume and os.path.exists(STATE_FILE):
        state = AgentState.load(STATE_FILE)
        logger.info("Resuming run from iteration %d, phase %s", state.iteration, state.phase)
        resume_msg = (
            f"Run resumed. Currently at iteration {state.iteration}, phase {state.phase}. "
            f"Best reward so far: {state.best_reward:+d}. Continue from where you left off."
        )
        state.conversation_history.append({"role": "user", "content": resume_msg})
    else:
        state = AgentState.new()
        logger.info("Starting fresh run")

    # Token budget
    budget = TokenBudget()
    budget.set_callbacks(
        alert=lambda msg: human_loop.notify(msg),
        halt=lambda: state.save(STATE_FILE),
    )

    # Wire up LLM client
    llm_client.register_tool_schemas(tool_module.TOOL_SCHEMAS)
    llm_client.set_token_callback(
        lambda inp, out, src: (
            budget.record(inp, out, src, state),
            state.record_tokens(inp, out, src),
        )
    )
    llm_client.set_human_escalation(
        lambda subject, body: (
            human_loop.send_alert(subject, body, state),
            human_loop.wait_for_reply(),
        )
    )

    # Seed conversation if fresh
    if not state.conversation_history:
        state.conversation_history.append({
            "role": "user",
            "content": (
                "Begin the kernel optimisation process. "
                "Start by querying RAGKernelCorpusTool for INT4 AWQ dequantization examples, "
                "then draft your first kernel implementation."
            ),
        })

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    stop_reason = "unknown"
    try:
        while True:
            # Time budget check
            if state.elapsed_hours() >= TIME_BUDGET_HOURS:
                stop_reason = f"24-hour time budget expired ({state.elapsed_hours():.2f}h)"
                break

            # Token budget check
            if budget.exhausted():
                stop_reason = "Token budget exhausted"
                break

            if state.phase == "done":
                stop_reason = "Phase C completed successfully"
                break

            if state.phase == "failed":
                stop_reason = "Agent reported unrecoverable failure"
                break

            if state.phase == "C":
                success = _run_phase_c(state)
                if not success:
                    # _run_phase_c already reverted state.phase to "A"
                    continue
                stop_reason = "Phase C completed successfully"
                break

            # Phase A step
            try:
                _run_phase_a_step(state, budget)
            except TokenBudgetExceededError:
                stop_reason = "Token budget exhausted mid-iteration"
                break
            except RuntimeError as exc:
                # LLM routing failure — human was already alerted inside llm_client
                logger.error("LLM routing failure: %s", exc)
                state.conversation_history.append({
                    "role": "user",
                    "content": f"[LLM ERROR] {exc}\nPlease provide guidance or the agent will retry.",
                })
                time.sleep(5)

    except KeyboardInterrupt:
        stop_reason = "Manual interrupt (Ctrl-C)"
    except Exception as exc:
        logger.exception("Unhandled exception in main loop")
        human_loop.send_alert(
            "Unhandled exception — agent halted",
            f"{type(exc).__name__}: {exc}",
            state,
        )
        stop_reason = f"Unhandled exception: {type(exc).__name__}"
    finally:
        state.save(STATE_FILE)
        _send_final_report(state, stop_reason, budget)
        logger.info("Agent stopped. Reason: %s", stop_reason)


if __name__ == "__main__":
    main()

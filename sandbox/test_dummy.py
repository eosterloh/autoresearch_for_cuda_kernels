"""
SandboxSpeedTestTool runner.

Executed as a subprocess by tools.py so GPU memory is isolated from the
orchestrator process. Outputs a single JSON line to stdout:
  {"latency_ms": <float>, "correctness_passed": <bool>}

Usage
-----
python test_dummy.py --kernel cuda_kernel --batch 1 --in_features 4096 --out_features 4096
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


def run_test(
    kernel_name: str,
    batch_size: int,
    in_features: int,
    out_features: int,
    n_warmup: int = 100,
    n_timed: int = 1000,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    device = torch.device("cuda")

    # ------------------------------------------------------------------ #
    # Load the compiled kernel extension                                   #
    # ------------------------------------------------------------------ #
    try:
        kernel_mod = importlib.import_module(kernel_name)
    except ModuleNotFoundError:
        build_dir = Path(__file__).parent / "build"
        sys.path.insert(0, str(build_dir))
        try:
            kernel_mod = importlib.import_module(kernel_name)
        except ModuleNotFoundError as exc:
            return {
                "latency_ms": 0.0,
                "correctness_passed": False,
                "error": f"Cannot import kernel '{kernel_name}': {exc}",
            }

    # ------------------------------------------------------------------ #
    # Synthesise AWQ-style INT4 inputs                                     #
    # ------------------------------------------------------------------ #
    # Weights packed: 8 INT4s per int32, lower nibble first.
    # Group size 128 — one scale + zero_point per 128 input channels.
    group_size = 128
    n_groups = in_features // group_size

    weight_packed = torch.randint(
        0, 2**31, (out_features, in_features // 8), dtype=torch.int32, device=device
    )
    scales = (torch.rand(out_features, n_groups, dtype=torch.float16, device=device) * 0.1 + 0.001)
    zeros  = torch.randint(0, 8, (out_features, n_groups), dtype=torch.float16, device=device)
    x      = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # ------------------------------------------------------------------ #
    # Reference: unpack → dequant → torch.mm                              #
    # ------------------------------------------------------------------ #
    weight_fp16 = torch.zeros(out_features, in_features, dtype=torch.float16, device=device)
    for bit in range(8):
        nibble = ((weight_packed >> (bit * 4)) & 0xF).to(torch.float16)
        weight_fp16[:, bit::8] = nibble

    weight_dequant = torch.zeros_like(weight_fp16)
    for g in range(n_groups):
        start, end = g * group_size, (g + 1) * group_size
        s = scales[:, g].unsqueeze(1)
        z = zeros[:, g].unsqueeze(1)
        weight_dequant[:, start:end] = (weight_fp16[:, start:end] - z) * s

    reference = torch.mm(x, weight_dequant.t())

    # ------------------------------------------------------------------ #
    # Kernel forward pass                                                  #
    # ------------------------------------------------------------------ #
    torch.cuda.synchronize()
    try:
        kernel_output = kernel_mod.dequantize_matmul(x, weight_packed, scales, zeros)
    except AttributeError:
        return {
            "latency_ms": 0.0,
            "correctness_passed": False,
            "error": f"Kernel module has no 'dequantize_matmul' function.",
        }
    except Exception as exc:
        return {
            "latency_ms": 0.0,
            "correctness_passed": False,
            "error": f"Kernel raised: {exc}",
        }
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Correctness                                                          #
    # ------------------------------------------------------------------ #
    correctness_passed = torch.allclose(
        kernel_output.float(), reference.float(), atol=atol, rtol=rtol
    )

    # ------------------------------------------------------------------ #
    # Timing                                                               #
    # ------------------------------------------------------------------ #
    for _ in range(n_warmup):
        kernel_mod.dequantize_matmul(x, weight_packed, scales, zeros)
    torch.cuda.synchronize()

    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for _ in range(n_timed):
        kernel_mod.dequantize_matmul(x, weight_packed, scales, zeros)
    t_end.record()
    torch.cuda.synchronize()

    latency_ms = t_start.elapsed_time(t_end) / n_timed

    return {
        "latency_ms": round(latency_ms, 6),
        "correctness_passed": bool(correctness_passed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel",       default="cuda_kernel")
    parser.add_argument("--batch",        type=int, default=1)
    parser.add_argument("--in_features",  type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=4096)
    args = parser.parse_args()

    result = run_test(
        kernel_name=args.kernel,
        batch_size=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
    )
    print(json.dumps(result))

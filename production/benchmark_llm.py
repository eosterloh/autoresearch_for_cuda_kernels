"""
Phase C benchmark: measure TTFT and TPS on the patched Gemma 3 12B AWQ model.

Used in two modes:
  --mode kernel   : benchmark the custom kernel (called after monkey_patch.py)
  --mode compile  : benchmark torch.compile baseline for BenchmarkTool

Outputs a single JSON line to stdout (tools.py parses this):
  {
    "ttft_ms": <float>,
    "tps": <float>,
    "speedup_ratio": <float>,          # only in --mode kernel
    "torch_compile_latency_ms": <float>,  # only in --mode kernel
    "kernel_latency_ms": <float>          # only in --mode kernel
  }

Also used standalone by BenchmarkTool with --mode kernel to compare
a compiled CUDA kernel's raw MatMul latency against torch.compile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sandbox" / "build"))


# ------------------------------------------------------------------ #
# Kernel vs torch.compile micro-benchmark (used by BenchmarkTool)     #
# ------------------------------------------------------------------ #

def benchmark_kernel_vs_compile(
    kernel_name: str = "cuda_kernel",
    batch_size: int = 1,
    in_features: int = 4096,
    out_features: int = 4096,
    n_warmup: int = 50,
    n_timed: int = 500,
) -> dict:
    import importlib

    device = torch.device("cuda")

    # ---- torch.compile baseline ----
    ref_linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).to(device)
    compiled   = torch.compile(ref_linear, mode="max-autotune")
    x          = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    for _ in range(n_warmup):
        compiled(x)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_timed):
        compiled(x)
    t1.record()
    torch.cuda.synchronize()
    compile_ms = t0.elapsed_time(t1) / n_timed

    # ---- Custom kernel ----
    try:
        kernel_mod = importlib.import_module(kernel_name)
    except ModuleNotFoundError:
        return {
            "speedup_ratio": 0.0,
            "torch_compile_latency_ms": compile_ms,
            "kernel_latency_ms": 0.0,
            "error": f"Kernel '{kernel_name}' not found",
        }

    group_size = 128
    n_groups   = in_features // group_size
    weight_packed = torch.randint(
        0, 2**31, (out_features, in_features // 8), dtype=torch.int32, device=device
    )
    scales = torch.rand(out_features, n_groups, dtype=torch.float16, device=device) * 0.1 + 0.001
    zeros  = torch.zeros(out_features, n_groups, dtype=torch.float16, device=device)

    for _ in range(n_warmup):
        kernel_mod.dequantize_matmul(x, weight_packed, scales, zeros)
    torch.cuda.synchronize()

    t2 = torch.cuda.Event(enable_timing=True)
    t3 = torch.cuda.Event(enable_timing=True)
    t2.record()
    for _ in range(n_timed):
        kernel_mod.dequantize_matmul(x, weight_packed, scales, zeros)
    t3.record()
    torch.cuda.synchronize()
    kernel_ms = t2.elapsed_time(t3) / n_timed

    speedup = compile_ms / kernel_ms if kernel_ms > 0 else 0.0
    return {
        "speedup_ratio": round(speedup, 4),
        "torch_compile_latency_ms": round(compile_ms, 6),
        "kernel_latency_ms": round(kernel_ms, 6),
    }


# ------------------------------------------------------------------ #
# Full TTFT + TPS benchmark on the patched model                       #
# ------------------------------------------------------------------ #

def benchmark_end_to_end(model_cache_key: str) -> dict:
    import tools as t
    from transformers import AutoTokenizer

    entry = t.get_cached_model(model_cache_key)
    if entry is None:
        return {"ttft_ms": 0.0, "tps": 0.0, "error": "Model not in cache"}

    model     = entry["model"]
    tokenizer = entry["tokenizer"]
    model_id  = entry["model_id"]

    model.eval()

    prompt  = "Explain the concept of CUDA shared memory in one paragraph."
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")

    # ---- TTFT (time to first token) ----
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_start) * 1000

    # ---- TPS over 128 tokens ----
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    tps = 128 / elapsed if elapsed > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 3),
        "tps": round(tps, 3),
    }


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kernel", "compile", "e2e"], default="kernel")
    parser.add_argument("--kernel",          default="cuda_kernel")
    parser.add_argument("--model_cache_key", default="")
    parser.add_argument("--batch",           type=int, default=1)
    parser.add_argument("--in_features",     type=int, default=4096)
    parser.add_argument("--out_features",    type=int, default=4096)
    args = parser.parse_args()

    if args.mode == "kernel":
        result = benchmark_kernel_vs_compile(
            kernel_name=args.kernel,
            batch_size=args.batch,
            in_features=args.in_features,
            out_features=args.out_features,
        )
    elif args.mode == "e2e":
        if not args.model_cache_key:
            print(json.dumps({"error": "--model_cache_key required for e2e mode"}))
            sys.exit(1)
        result = benchmark_end_to_end(args.model_cache_key)
    else:
        # compile-only timing
        result = benchmark_kernel_vs_compile(
            kernel_name=args.kernel,
            batch_size=args.batch,
            in_features=args.in_features,
            out_features=args.out_features,
        )

    print(json.dumps(result))


if __name__ == "__main__":
    main()

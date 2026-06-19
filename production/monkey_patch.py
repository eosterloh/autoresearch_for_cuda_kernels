"""
Phase C: monkey-patch all nn.Linear layers in Gemma 3 12B AWQ with the
compiled custom CUDA dequantization kernel.

Usage (called by orchestrator as subprocess or imported)
--------------------------------------------------------
python monkey_patch.py --model_cache_key <uuid> --benchmark

The model must already be loaded into _MODEL_CACHE by TransformersWeightLoaderTool.
When run as a subprocess from the project root, it imports tools.py to access
the cache.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow importing from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sandbox" / "build"))


# ------------------------------------------------------------------ #
# Custom linear layer that calls our CUDA kernel                       #
# ------------------------------------------------------------------ #

class AgenticLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that routes computation through the
    compiled INT4 AWQ dequantize + matmul CUDA kernel.

    Falls back to standard FP16 torch.mm if the kernel is unavailable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        kernel_module=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self._kernel = kernel_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        if self._kernel is not None:
            out = self._kernel.dequantize_matmul(
                x_2d.to(torch.float16),
                self.weight_packed,
                self.scales,
                self.zeros,
            )
        else:
            # Fallback: standard matmul (correctness reference)
            out = torch.mm(x_2d.to(torch.float16), self._dequant_weight().t())

        if self.bias is not None:
            out = out + self.bias

        return out.view(*original_shape[:-1], self.out_features)

    def _dequant_weight(self) -> torch.Tensor:
        """Fallback dequant for when kernel is not available."""
        group_size = 128
        n_groups = self.in_features // group_size
        weight_fp16 = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float16, device=self.weight_packed.device,
        )
        for bit in range(8):
            nibble = ((self.weight_packed >> (bit * 4)) & 0xF).to(torch.float16)
            weight_fp16[:, bit::8] = nibble

        weight_dequant = torch.zeros_like(weight_fp16)
        for g in range(n_groups):
            start, end = g * group_size, (g + 1) * group_size
            s = self.scales[:, g].unsqueeze(1)
            z = self.zeros[:, g].unsqueeze(1)
            weight_dequant[:, start:end] = (weight_fp16[:, start:end] - z) * s
        return weight_dequant


# ------------------------------------------------------------------ #
# Patching logic                                                       #
# ------------------------------------------------------------------ #

def _get_parent_and_attr(model: nn.Module, full_name: str):
    """Navigate the module tree to return (parent_module, attr_name)."""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def patch_model(
    model: nn.Module,
    kernel_module=None,
    group_size: int = 128,
) -> tuple[int, list[str]]:
    """
    Replace all nn.Linear layers with AgenticLinear.

    Returns (patch_count, list_of_patched_layer_names).
    """
    patched = []
    # Collect targets before iterating to avoid mutation during traversal
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    for name, module in targets:
        in_f  = module.in_features
        out_f = module.out_features
        n_groups = in_f // group_size

        # AWQ models store weights as int32 packed tensors already.
        # If this is a standard FP16 linear (e.g. lm_head), pack it naively.
        w = module.weight.data  # [out, in]

        if w.dtype == torch.float16:
            # Pack FP16 weights as trivial INT4 (zero-point=8, scale=1/16)
            # so AgenticLinear can still call the kernel for testing.
            # The agent will replace this with proper AWQ unpacking.
            w_int4 = (w * 16).clamp(0, 15).to(torch.int32)
            weight_packed = torch.zeros(
                out_f, in_f // 8, dtype=torch.int32, device=w.device
            )
            for bit in range(8):
                weight_packed |= (w_int4[:, bit::8] & 0xF) << (bit * 4)
            scales = torch.ones(out_f, n_groups, dtype=torch.float16, device=w.device) / 16.0
            zeros  = torch.full((out_f, n_groups), 8.0, dtype=torch.float16, device=w.device)
        elif w.dtype == torch.int32:
            # Already packed (AWQ format)
            weight_packed = w
            # Scales/zeros should be stored as separate buffers in AWQ models
            scales = getattr(module, "scales", torch.ones(out_f, n_groups, dtype=torch.float16, device=w.device))
            zeros  = getattr(module, "zeros",  torch.zeros(out_f, n_groups, dtype=torch.float16, device=w.device))
        else:
            # Skip unsupported dtypes (e.g. bfloat16)
            continue

        bias = module.bias.data if module.bias is not None else None

        new_layer = AgenticLinear(
            in_features=in_f,
            out_features=out_f,
            weight_packed=weight_packed,
            scales=scales,
            zeros=zeros,
            bias=bias,
            kernel_module=kernel_module,
        )

        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, new_layer)
        patched.append(name)

    return len(patched), patched


# ------------------------------------------------------------------ #
# CLI entry point                                                      #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Monkey-patch Gemma 3 12B AWQ with custom kernel")
    parser.add_argument("--model_cache_key", required=True,
                        help="UUID key from TransformersWeightLoaderTool")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark_llm after patching")
    args = parser.parse_args()

    # Load kernel module
    try:
        kernel_mod = importlib.import_module("cuda_kernel")
        print(f"[monkey_patch] Loaded kernel module: cuda_kernel", flush=True)
    except ModuleNotFoundError:
        kernel_mod = None
        print("[monkey_patch] WARNING: cuda_kernel not found — using FP16 fallback", flush=True)

    # Retrieve model from cache
    import tools as t
    entry = t.get_cached_model(args.model_cache_key)
    if entry is None:
        print(json.dumps({"success": False, "error": "Model not found in cache"}))
        sys.exit(1)

    model = entry["model"]
    model.eval()

    # Patch
    count, patched_names = patch_model(model, kernel_module=kernel_mod)
    print(f"[monkey_patch] Patched {count} nn.Linear layers", flush=True)
    print(f"[monkey_patch] First 5: {patched_names[:5]}", flush=True)

    if args.benchmark:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "production" / "benchmark_llm.py"),
             "--model_cache_key", args.model_cache_key],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(json.dumps({"success": False, "error": result.stderr}))
            sys.exit(1)
    else:
        print(json.dumps({"success": True, "patched_count": count}))


if __name__ == "__main__":
    main()

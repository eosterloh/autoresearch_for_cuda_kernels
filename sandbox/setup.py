"""
JIT compilation script for the CUDA kernel extension.
Used by SyntaxCompileTool via torch.utils.cpp_extension.load().
Can also be run directly: python setup.py build_ext --inplace
"""

from torch.utils.cpp_extension import load
from pathlib import Path

HERE = Path(__file__).parent

cuda_kernel = load(
    name="cuda_kernel",
    sources=[
        str(HERE / "binding.cpp"),
        str(HERE / "current_kernel.cu"),
    ],
    extra_cuda_cflags=[
        "-arch=sm_100",      # GB10 Grace Blackwell
        "--use_fast_math",
        "-O3",
        "-std=c++17",
        "-lineinfo",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    build_directory=str(HERE / "build"),
    verbose=True,
)

print("Compilation successful.")

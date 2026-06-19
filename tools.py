"""
All 7 tool implementations exposed to the LLM agent, plus a dispatcher.

Security notes
--------------
- CodeWriterTool: file_path is validated to stay within the sandbox/ directory.
- BashExecutionTool: working_dir is validated to stay within the project root.
- TransformersWeightLoaderTool: model objects are stored in a module-level cache
  keyed by UUID strings — no raw memory pointers are exposed to the LLM.

Tool call schema (what the LLM must send)
-----------------------------------------
{
  "tool_call": {
    "name": "<ToolName>",
    "arguments": { ... }
  }
}
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).parent.resolve()
SANDBOX_DIR = PROJECT_ROOT / "sandbox"

# ------------------------------------------------------------------ #
# Model cache (TransformersWeightLoaderTool)                           #
# ------------------------------------------------------------------ #

_MODEL_CACHE: dict[str, Any] = {}

# ------------------------------------------------------------------ #
# Path validation helpers                                              #
# ------------------------------------------------------------------ #

def _safe_sandbox_path(file_path: str) -> Path:
    """Resolve and validate that file_path stays inside sandbox/."""
    target = (SANDBOX_DIR / file_path).resolve()
    if not str(target).startswith(str(SANDBOX_DIR)):
        raise ValueError(
            f"Path traversal rejected: '{file_path}' resolves outside sandbox/"
        )
    return target


def _safe_working_dir(working_dir: str) -> Path:
    """Resolve and validate that working_dir stays inside the project root."""
    target = (PROJECT_ROOT / working_dir).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        raise ValueError(
            f"Path traversal rejected: '{working_dir}' resolves outside project root"
        )
    if not target.exists():
        raise ValueError(f"working_dir does not exist: {target}")
    return target


# ------------------------------------------------------------------ #
# 1. CodeWriterTool                                                    #
# ------------------------------------------------------------------ #

def code_writer(file_path: str, content: str, mode: str = "overwrite") -> dict:
    """
    Write raw C++, CUDA, or Python code to the sandbox directory.
    mode: "overwrite" | "append"
    """
    if mode not in ("overwrite", "append"):
        return {"status": "fail", "error": f"Invalid mode '{mode}'. Use 'overwrite' or 'append'."}
    try:
        target = _safe_sandbox_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        open_mode = "w" if mode == "overwrite" else "a"
        with open(target, open_mode, encoding="utf-8") as f:
            f.write(content)
        logger.info("CodeWriterTool: wrote %d chars to %s (%s)", len(content), target, mode)
        return {"status": "success", "path": str(target)}
    except (ValueError, OSError) as exc:
        return {"status": "fail", "error": str(exc)}


# ------------------------------------------------------------------ #
# 2. BashExecutionTool                                                 #
# ------------------------------------------------------------------ #

def bash_execution(
    command: str,
    working_dir: str = ".",
    timeout_seconds: int = 300,
) -> dict:
    """
    Execute a shell command. Returns stdout, stderr, exit_code.
    Processes that exceed timeout_seconds are killed.
    """
    try:
        cwd = _safe_working_dir(working_dir)
    except ValueError as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": -1}

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-8000:] if len(result.stderr) > 8000 else result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"TIMEOUT: process killed after {timeout_seconds}s",
            "exit_code": -1,
        }
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": -1}


# ------------------------------------------------------------------ #
# 3. SyntaxCompileTool                                                 #
# ------------------------------------------------------------------ #

def syntax_compile(
    cu_file_path: str = "current_kernel.cu",
    cpp_file_path: str = "binding.cpp",
) -> dict:
    """
    Compile the CUDA extension via torch.utils.cpp_extension.load().
    Returns compiler_traceback and success bool.
    nvcc flags target GB10 Grace Blackwell (sm_100).
    """
    try:
        cu_path = _safe_sandbox_path(cu_file_path)
        cpp_path = _safe_sandbox_path(cpp_file_path)
    except ValueError as exc:
        return {"compiler_traceback": str(exc), "success": False}

    if not cu_path.exists():
        return {"compiler_traceback": f"File not found: {cu_path}", "success": False}
    if not cpp_path.exists():
        return {"compiler_traceback": f"File not found: {cpp_path}", "success": False}

    # Capture all compilation output
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        import torch.utils.cpp_extension as cpp_ext

        # Unload any previously compiled version
        if "cuda_kernel" in sys.modules:
            del sys.modules["cuda_kernel"]

        build_dir = str(SANDBOX_DIR / "build")
        os.makedirs(build_dir, exist_ok=True)

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            cpp_ext.load(
                name="cuda_kernel",
                sources=[str(cpp_path), str(cu_path)],
                extra_cuda_cflags=[
                    "-arch=sm_100",
                    "--use_fast_math",
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                ],
                extra_cflags=["-O3", "-std=c++17"],
                build_directory=build_dir,
                verbose=True,
            )

        traceback = stdout_buf.getvalue() + stderr_buf.getvalue()
        logger.info("SyntaxCompileTool: compilation succeeded")
        return {"compiler_traceback": traceback, "success": True}

    except Exception as exc:
        traceback = stdout_buf.getvalue() + stderr_buf.getvalue() + f"\n{exc}"
        logger.warning("SyntaxCompileTool: compilation failed: %.300s", exc)
        return {"compiler_traceback": traceback, "success": False}


# ------------------------------------------------------------------ #
# 4. SandboxSpeedTestTool                                              #
# ------------------------------------------------------------------ #

def sandbox_speed_test(
    kernel_name: str = "cuda_kernel",
    batch_size: int = 1,
    in_features: int = 4096,
    out_features: int = 4096,
) -> dict:
    """
    Run test_dummy.py as a subprocess to isolate GPU memory from the orchestrator.
    Returns latency_ms and correctness_passed.
    """
    script = str(SANDBOX_DIR / "test_dummy.py")
    if not os.path.exists(script):
        return {"latency_ms": 0.0, "correctness_passed": False, "error": "test_dummy.py not found"}

    cmd = (
        f"{sys.executable} {script} "
        f"--kernel {kernel_name} "
        f"--batch {batch_size} "
        f"--in_features {in_features} "
        f"--out_features {out_features}"
    )
    result = bash_execution(cmd, working_dir="sandbox", timeout_seconds=120)

    if result["exit_code"] != 0:
        return {
            "latency_ms": 0.0,
            "correctness_passed": False,
            "error": result["stderr"] or result["stdout"],
        }

    try:
        data = json.loads(result["stdout"].strip())
        return {
            "latency_ms": float(data["latency_ms"]),
            "correctness_passed": bool(data["correctness_passed"]),
        }
    except (json.JSONDecodeError, KeyError) as exc:
        return {
            "latency_ms": 0.0,
            "correctness_passed": False,
            "error": f"Failed to parse test_dummy output: {exc}\n{result['stdout']}",
        }


# ------------------------------------------------------------------ #
# 5. BenchmarkTool                                                     #
# ------------------------------------------------------------------ #

def benchmark(kernel_name: str = "cuda_kernel") -> dict:
    """
    Compare custom kernel vs torch.compile baseline.
    Returns speedup_ratio = torch_compile_latency / kernel_latency.
    """
    script = str(PROJECT_ROOT / "production" / "benchmark_llm.py")
    cmd = f"{sys.executable} {script} --mode kernel --kernel {kernel_name}"
    result = bash_execution(cmd, working_dir=".", timeout_seconds=180)

    if result["exit_code"] != 0:
        return {
            "speedup_ratio": 0.0,
            "torch_compile_latency_ms": 0.0,
            "kernel_latency_ms": 0.0,
            "error": result["stderr"] or result["stdout"],
        }

    try:
        data = json.loads(result["stdout"].strip())
        return {
            "speedup_ratio": float(data["speedup_ratio"]),
            "torch_compile_latency_ms": float(data["torch_compile_latency_ms"]),
            "kernel_latency_ms": float(data["kernel_latency_ms"]),
        }
    except (json.JSONDecodeError, KeyError) as exc:
        return {
            "speedup_ratio": 0.0,
            "torch_compile_latency_ms": 0.0,
            "kernel_latency_ms": 0.0,
            "error": f"Failed to parse benchmark output: {exc}\n{result['stdout']}",
        }


# ------------------------------------------------------------------ #
# 6. RAGKernelCorpusTool                                               #
# ------------------------------------------------------------------ #

def rag_kernel_corpus(search_query: str) -> dict:
    """
    Embed the query with nomic-embed-text via Ollama and query ChromaDB.
    Returns top-3 code snippets from the local CUDA kernel corpus.
    """
    import requests as req

    # Embed the query
    try:
        embed_resp = req.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": search_query},
            timeout=30,
        )
        embed_resp.raise_for_status()
        embedding = embed_resp.json()["embedding"]
    except Exception as exc:
        return {"code_snippets": f"[RAG embedding failed: {exc}]"}

    # Query ChromaDB
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "rag_corpus" / "db"))
        collection = client.get_or_create_collection("cuda_kernels")

        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            include=["documents", "metadatas"],
        )

        snippets = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            source = meta.get("source", "unknown")
            func = meta.get("function_name", "")
            snippets.append(f"--- Snippet {i+1} | {source} | {func} ---\n{doc}")

        return {"code_snippets": "\n\n".join(snippets) if snippets else "[No results found]"}

    except Exception as exc:
        return {"code_snippets": f"[RAG query failed: {exc}]"}


# ------------------------------------------------------------------ #
# 7. TransformersWeightLoaderTool                                      #
# ------------------------------------------------------------------ #

def transformers_weight_loader(
    model_id: str = "TheBloke/gemma-3-12B-AWQ",
    target_layer: str = "model.layers.0.self_attn.q_proj",
) -> dict:
    """
    Load the target model into VRAM. Store in module-level cache.
    Returns a UUID cache key (not a raw memory pointer).
    """
    import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return {"memory_pointer": "", "success": False, "error": str(exc)}

    cache_key = str(uuid.uuid4())
    try:
        logger.info("Loading model %s into VRAM...", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _MODEL_CACHE[cache_key] = {"model": model, "tokenizer": tokenizer, "model_id": model_id}
        logger.info("Model loaded. Cache key: %s", cache_key)
        return {"memory_pointer": cache_key, "success": True}

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"memory_pointer": "", "success": False, "error": f"GPU OOM: {exc}"}
    except Exception as exc:
        return {"memory_pointer": "", "success": False, "error": str(exc)}


def get_cached_model(cache_key: str) -> dict | None:
    """Retrieve a previously loaded model by its cache key."""
    return _MODEL_CACHE.get(cache_key)


def release_model(cache_key: str) -> None:
    """Free VRAM for a cached model."""
    import torch
    entry = _MODEL_CACHE.pop(cache_key, None)
    if entry:
        del entry["model"]
        torch.cuda.empty_cache()
        logger.info("Released model %s from VRAM", cache_key)


# ------------------------------------------------------------------ #
# Dispatcher                                                           #
# ------------------------------------------------------------------ #

TOOL_REGISTRY: dict[str, callable] = {
    "CodeWriterTool": code_writer,
    "BashExecutionTool": bash_execution,
    "SyntaxCompileTool": syntax_compile,
    "SandboxSpeedTestTool": sandbox_speed_test,
    "BenchmarkTool": benchmark,
    "RAGKernelCorpusTool": rag_kernel_corpus,
    "TransformersWeightLoaderTool": transformers_weight_loader,
}


def dispatch(tool_name: str, arguments: dict) -> dict:
    """Route a tool_call to the correct implementation."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY)}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {tool_name}: {exc}"}
    except Exception as exc:
        logger.exception("Unhandled error in %s", tool_name)
        return {"error": f"Tool {tool_name} raised: {exc}"}


# ------------------------------------------------------------------ #
# JSON schemas for LLM system prompt                                   #
# ------------------------------------------------------------------ #

TOOL_SCHEMAS = [
    {
        "name": "CodeWriterTool",
        "description": "Write raw C++, CUDA, or Python code to the sandbox directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path within sandbox/"},
                "content": {"type": "string", "description": "Full file content to write"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "BashExecutionTool",
        "description": "Execute shell commands (nvcc, ninja, pip, etc.). Timeout is strictly enforced.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "integer", "default": 300},
            },
            "required": ["command"],
        },
    },
    {
        "name": "SyntaxCompileTool",
        "description": "Compile the CUDA kernel via torch.utils.cpp_extension. Returns compiler errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "cu_file_path": {"type": "string", "default": "current_kernel.cu"},
                "cpp_file_path": {"type": "string", "default": "binding.cpp"},
            },
        },
    },
    {
        "name": "SandboxSpeedTestTool",
        "description": "Run the compiled kernel on random tensors. Returns latency_ms and correctness_passed.",
        "parameters": {
            "type": "object",
            "properties": {
                "kernel_name": {"type": "string", "default": "cuda_kernel"},
                "batch_size": {"type": "integer", "default": 1},
                "in_features": {"type": "integer", "default": 4096},
                "out_features": {"type": "integer", "default": 4096},
            },
        },
    },
    {
        "name": "BenchmarkTool",
        "description": "Compare custom kernel vs torch.compile. Returns speedup_ratio (>1.0 means faster).",
        "parameters": {
            "type": "object",
            "properties": {
                "kernel_name": {"type": "string", "default": "cuda_kernel"},
            },
        },
    },
    {
        "name": "RAGKernelCorpusTool",
        "description": "Search the local CUDA kernel corpus for architectural inspiration.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {"type": "string"},
            },
            "required": ["search_query"],
        },
    },
    {
        "name": "TransformersWeightLoaderTool",
        "description": "Load INT4 AWQ model weights into VRAM for Phase C testing. Returns a cache key.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "default": "TheBloke/gemma-3-12B-AWQ"},
                "target_layer": {"type": "string", "default": "model.layers.0.self_attn.q_proj"},
            },
        },
    },
]

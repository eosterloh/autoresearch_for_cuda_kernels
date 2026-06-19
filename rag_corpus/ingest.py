"""
One-time RAG corpus seeding script.

Sources
-------
1. CUDA-Agent-Ops-6K dataset from HuggingFace Hub
   (referenced in the CUDA Agent paper: arxiv 2602.24286v1)
2. Dao-AILab/flash-attention — reference INT4/FP16 kernels
3. vllm-project/vllm        — production AWQ kernel implementations

All .cu / .cuh files are chunked at __global__ / __device__ function
boundaries, embedded with nomic-embed-text via Ollama, and upserted into
a persistent ChromaDB collection named "cuda_kernels".

Run once before starting orchestrator.py:
  python rag_corpus/ingest.py

Optional flags:
  --skip-hf           Skip HuggingFace dataset download
  --skip-github       Skip GitHub repo cloning
  --corpus-path PATH  Override default rag_corpus/db path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).parent.parent
DB_PATH        = PROJECT_ROOT / "rag_corpus" / "db"
OLLAMA_BASE    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL    = "nomic-embed-text"
COLLECTION     = "cuda_kernels"
BATCH_SIZE     = 50   # upsert batch size for ChromaDB

GITHUB_REPOS = [
    "https://github.com/Dao-AILab/flash-attention.git",
    "https://github.com/vllm-project/vllm.git",
]

HF_DATASET = "cuda-agent/CUDA-Agent-Ops-6K"   # adjust if the actual HF repo differs


# ------------------------------------------------------------------ #
# Chunking                                                             #
# ------------------------------------------------------------------ #

_FUNC_BOUNDARY = re.compile(
    r"(?:^|\n)(?:__global__|__device__|__host__|extern \"C\")\s+[\w\s\*<>:,]+\(",
    re.MULTILINE,
)


def chunk_cu_file(path: Path, source_label: str) -> list[dict]:
    """
    Split a .cu / .cuh file into function-level chunks.
    Returns list of {"id": str, "text": str, "metadata": dict}.
    """
    try:
        code = path.read_text(errors="replace")
    except OSError:
        return []

    # Find all function start positions
    boundaries = [m.start() for m in _FUNC_BOUNDARY.finditer(code)]
    if not boundaries:
        # File has no __global__/__device__ — include whole file as one chunk
        if len(code) > 100:
            return [_make_chunk(code, path, source_label, "file_level")]
        return []

    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(code)
        fragment = code[start:end].strip()
        if len(fragment) < 50:
            continue
        # Extract function name from the first line
        first_line = fragment.split("\n")[0]
        func_match = re.search(r"(\w+)\s*\(", first_line)
        func_name = func_match.group(1) if func_match else f"func_{i}"
        chunks.append(_make_chunk(fragment, path, source_label, func_name))

    return chunks


def _make_chunk(text: str, path: Path, source: str, func_name: str) -> dict:
    uid = hashlib.sha256(f"{source}:{path}:{func_name}:{text[:64]}".encode()).hexdigest()[:16]
    return {
        "id": uid,
        "text": text[:4000],  # ChromaDB document limit
        "metadata": {
            "source": source,
            "file_path": str(path),
            "function_name": func_name,
        },
    }


# ------------------------------------------------------------------ #
# Embedding via Ollama                                                 #
# ------------------------------------------------------------------ #

def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using nomic-embed-text via Ollama."""
    embeddings = []
    for text in texts:
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        except Exception as exc:
            logger.warning("Embedding failed for chunk: %s", exc)
            embeddings.append(None)
        time.sleep(0.05)  # gentle rate limit
    return embeddings


# ------------------------------------------------------------------ #
# ChromaDB upsert                                                      #
# ------------------------------------------------------------------ #

def upsert_chunks(collection, chunks: list[dict]) -> int:
    """Embed and upsert a batch of chunks. Returns count upserted."""
    texts  = [c["text"] for c in chunks]
    embeds = embed(texts)

    valid_ids   = []
    valid_docs  = []
    valid_embeds= []
    valid_metas = []

    for chunk, emb in zip(chunks, embeds):
        if emb is None:
            continue
        valid_ids.append(chunk["id"])
        valid_docs.append(chunk["text"])
        valid_embeds.append(emb)
        valid_metas.append(chunk["metadata"])

    if not valid_ids:
        return 0

    collection.upsert(
        ids=valid_ids,
        documents=valid_docs,
        embeddings=valid_embeds,
        metadatas=valid_metas,
    )
    return len(valid_ids)


# ------------------------------------------------------------------ #
# Source 1: CUDA-Agent-Ops-6K (HuggingFace)                           #
# ------------------------------------------------------------------ #

def ingest_hf_dataset(collection) -> int:
    logger.info("Ingesting CUDA-Agent-Ops-6K from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed — skipping HF dataset")
        return 0

    try:
        local_dir = snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )
    except Exception as exc:
        logger.warning("HF dataset download failed: %s", exc)
        return 0

    total = 0
    batch: list[dict] = []
    for cu_file in Path(local_dir).rglob("*.cu"):
        chunks = chunk_cu_file(cu_file, "cuda-agent-ops-6k")
        batch.extend(chunks)
        if len(batch) >= BATCH_SIZE:
            total += upsert_chunks(collection, batch)
            logger.info("HF: %d chunks ingested so far", total)
            batch = []
    if batch:
        total += upsert_chunks(collection, batch)

    logger.info("HF dataset: %d chunks ingested", total)
    return total


# ------------------------------------------------------------------ #
# Source 2: GitHub repos (flash-attention, vLLM)                      #
# ------------------------------------------------------------------ #

def ingest_github_repos(collection) -> int:
    total = 0
    tmp_base = tempfile.mkdtemp(prefix="autoresearch_rag_")
    try:
        for repo_url in GITHUB_REPOS:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            clone_dir = Path(tmp_base) / repo_name
            logger.info("Cloning %s (shallow)...", repo_url)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--no-checkout", repo_url, str(clone_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning("Clone failed for %s: %s", repo_url, result.stderr[:200])
                continue

            # Sparse checkout only .cu and .cuh files
            subprocess.run(
                ["git", "-C", str(clone_dir), "sparse-checkout", "set", "*.cu", "*.cuh"],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout"],
                capture_output=True, timeout=60,
            )

            batch: list[dict] = []
            for ext in ("*.cu", "*.cuh"):
                for cu_file in clone_dir.rglob(ext):
                    chunks = chunk_cu_file(cu_file, repo_name)
                    batch.extend(chunks)
                    if len(batch) >= BATCH_SIZE:
                        total += upsert_chunks(collection, batch)
                        logger.info("%s: %d chunks ingested so far", repo_name, total)
                        batch = []
            if batch:
                total += upsert_chunks(collection, batch)

            logger.info("%s: done", repo_name)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    logger.info("GitHub repos: %d total chunks ingested", total)
    return total


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the RAG corpus for autoresearch-kernel")
    parser.add_argument("--skip-hf",     action="store_true", help="Skip HuggingFace dataset")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub repo cloning")
    parser.add_argument("--corpus-path", default=str(DB_PATH),
                        help="Path to ChromaDB persistent storage")
    args = parser.parse_args()

    # Verify Ollama is reachable
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(EMBED_MODEL in m for m in models):
            logger.warning(
                "%s not found in Ollama. Pull it first: ollama pull %s",
                EMBED_MODEL, EMBED_MODEL,
            )
    except Exception as exc:
        logger.error("Cannot reach Ollama at %s: %s", OLLAMA_BASE, exc)
        sys.exit(1)

    # Init ChromaDB
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=args.corpus_path)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' ready at %s", COLLECTION, args.corpus_path)
    logger.info("Existing documents: %d", collection.count())

    grand_total = 0

    if not args.skip_hf:
        grand_total += ingest_hf_dataset(collection)

    if not args.skip_github:
        grand_total += ingest_github_repos(collection)

    logger.info("Ingestion complete. Total chunks added: %d", grand_total)
    logger.info("Collection now has %d documents", collection.count())


if __name__ == "__main__":
    main()

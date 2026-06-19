# autoresearch-kernel

Autonomous LLM-driven agent that generates, compiles, and optimises a custom
CUDA INT4 → FP16 dequantization MatMul kernel on a DGX Spark (GB10 Grace
Blackwell, `sm_100`). Uses a reinforcement learning loop to beat `torch.compile`
and then validates the kernel in a live Gemma 3 12B AWQ inference pass.

---

## Architecture

```
orchestrator.py           Main RL loop (Phase A → B gate → Phase C)
llm_client.py             Ollama (Nemotron 30B) primary / Anthropic fallback
tools.py                  7 tools exposed to the LLM agent
reward_system.py          -1 / +1 / +3 reward + Phase B gate
state.py                  Atomic state persistence (resume on crash)
token_budget.py           Per-session token tracking
human_loop.py             Telegram block-and-wait escalation

sandbox/
  current_kernel.cu       Active CUDA kernel (rewritten by the agent)
  binding.cpp             pybind11 PyTorch wrapper
  test_dummy.py           Correctness + latency benchmark (runs in subprocess)
  setup.py                torch.utils.cpp_extension JIT builder

production/
  monkey_patch.py         Replace nn.Linear layers with AgenticLinear
  benchmark_llm.py        Measure TTFT and TPS on Gemma 3 12B AWQ

rag_corpus/
  ingest.py               Seed ChromaDB from CUDA-Agent-Ops-6K + GitHub
  db/                     ChromaDB persistent storage
```

---

## Required API Keys

| Service | Purpose | Where to get it |
|---|---|---|
| **Anthropic** | Smart LLM fallback (Claude Sonnet) | https://console.anthropic.com/settings/keys |
| **Telegram Bot** | Human-in-the-loop alerts | @BotFather on Telegram → `/newbot` |
| **HuggingFace Hub** | Gemma 3 12B AWQ download + corpus | https://huggingface.co/settings/tokens |
| **GitHub** | RAG corpus seeding (flash-attention, vLLM) | https://github.com/settings/tokens |

---

## Setup

### 1. Python environment

```bash
cd autoresearch-kernel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in all four API keys
source .env
```

### 3. Ollama — start the server and pull the two models

Ollama is a native binary, not Docker. It must be running before the agent starts.

```bash
# Start Ollama in a dedicated tmux window (keeps running after SSH disconnect)
tmux new-window -n ollama
ollama serve
# You should see: Listening on 127.0.0.1:11434
```

Then in another window, pull the models:
```bash
ollama pull nemotron-nano          # orchestrator LLM (~30B, primary)
ollama pull nomic-embed-text       # RAG embedding model
```

To verify Ollama is up at any time:
```bash
curl http://localhost:11434/api/tags
```

### 4. HuggingFace — accept the Gemma 3 licence

Visit https://huggingface.co/google/gemma-3-12b and click **Agree and access repository**.
Then authenticate:
```bash
huggingface-cli login              # paste your HF_TOKEN when prompted
```

### 5. Telegram — get your chat ID

After creating the bot with @BotFather and adding the token to `.env`:
```bash
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates"
# Send any message to your bot first, then run this.
# Copy the numeric "id" from result[0].message.chat.id → TELEGRAM_CHAT_ID
```

### 6. Seed the RAG corpus (one-time, ~30–60 min)

```bash
python rag_corpus/ingest.py
# Skip GitHub cloning if bandwidth is limited:
# python rag_corpus/ingest.py --skip-github
```

---

## Running

```bash
# Fresh run (24-hour autonomous loop)
python orchestrator.py

# Resume from a previous state.json
python orchestrator.py --resume
```

The agent runs fully autonomously. You will receive Telegram messages:
- **80% token budget** — non-blocking alert
- **Human needed** — agent blocks and waits for your reply
- **Run complete** — final summary with best kernel path and Phase C results

---

## Reward system

| Signal | Condition |
|---|---|
| **-1** | Compile failure or `torch.allclose` check failed |
| **+1** | Correct output but slower than `torch.compile` |
| **+3** | Correct output and faster than `torch.compile` |

**Phase B gate**: +3 across 3 consecutive evaluations at batch sizes `[1, 32, 128]`.
Any non-+3 resets the counter.

---

## Dry run / smoke test

```bash
# Cap at 3 minutes and 10k tokens to test the full loop end-to-end
MAX_SESSION_TOKENS=10000 TIME_BUDGET_HOURS=0.05 python orchestrator.py
```

---

## nvcc flags

All kernels compile with:
```
-arch=sm_100 --use_fast_math -O3 -std=c++17 -lineinfo
```

`sm_100` targets the GB10 Grace Blackwell Superchip.

---

## Resuming after a crash

```bash
python orchestrator.py --resume
```

State is saved atomically to `state.json` after every iteration and before
every Telegram escalation, so no work is lost.

"""
LLM client with Ollama-primary / Anthropic-fallback routing.

Ollama path
-----------
Posts to http://localhost:11434/api/chat with format="json".
Expects the model to return a JSON object with a top-level "tool_call" key:
  {
    "tool_call": {
      "name": "<tool name>",
      "arguments": { ... }
    }
  }

On JSONDecodeError or missing tool_call: retries up to MAX_OLLAMA_RETRIES with
a corrective system injection. After exhausting retries, escalates to Anthropic.

Anthropic path
--------------
Uses the native tool-use API (claude-opus-4-8). Anthropic enforces structured
tool-call responses natively, so no JSON forcing is needed.

If Anthropic also fails (API error / network): escalates to human_loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

MAX_OLLAMA_RETRIES = 3
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nemotron-nano")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


# ------------------------------------------------------------------ #
# Tool schema registry (populated by orchestrator at start-up)        #
# ------------------------------------------------------------------ #

_TOOL_SCHEMAS: list[dict] = []


def register_tool_schemas(schemas: list[dict]) -> None:
    global _TOOL_SCHEMAS
    _TOOL_SCHEMAS = schemas


# ------------------------------------------------------------------ #
# Token callback (injected by orchestrator)                            #
# ------------------------------------------------------------------ #

_token_callback: Optional[Callable[[int, int, str], None]] = None


def set_token_callback(cb: Callable[[int, int, str], None]) -> None:
    global _token_callback
    _token_callback = cb


def _record_tokens(inp: int, out: int, source: str) -> None:
    if _token_callback:
        _token_callback(inp, out, source)


# ------------------------------------------------------------------ #
# Human-loop escalation callback                                       #
# ------------------------------------------------------------------ #

_human_escalate: Optional[Callable[[str, str], None]] = None


def set_human_escalation(cb: Callable[[str, str], None]) -> None:
    global _human_escalate
    _human_escalate = cb


# ------------------------------------------------------------------ #
# Ollama                                                               #
# ------------------------------------------------------------------ #

def _ollama_call(messages: list[dict]) -> dict:
    """
    Returns a parsed tool_call dict: {"name": str, "arguments": dict}
    Raises RuntimeError after MAX_OLLAMA_RETRIES failed parse attempts.
    """
    working_messages = list(messages)

    for attempt in range(1, MAX_OLLAMA_RETRIES + 1):
        payload = {
            "model": OLLAMA_MODEL,
            "messages": working_messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Ollama request error (attempt %d): %s", attempt, exc)
            if attempt == MAX_OLLAMA_RETRIES:
                raise RuntimeError(f"Ollama unavailable after {attempt} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        body = resp.json()
        raw_content = body.get("message", {}).get("content", "")

        # Approximate token counts from Ollama eval metrics
        prompt_tokens = body.get("prompt_eval_count", len(str(working_messages)) // 4)
        completion_tokens = body.get("eval_count", len(raw_content) // 4)
        _record_tokens(prompt_tokens, completion_tokens, "ollama")

        # Parse and validate
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning(
                "Ollama response not valid JSON (attempt %d). Raw: %.200s",
                attempt, raw_content,
            )
            parsed = None

        if parsed and isinstance(parsed.get("tool_call"), dict):
            tc = parsed["tool_call"]
            if "name" in tc and "arguments" in tc:
                return tc

        # Inject correction for next attempt
        working_messages = working_messages + [
            {"role": "assistant", "content": raw_content or ""},
            {
                "role": "user",
                "content": (
                    "Your last response was not valid JSON or did not contain a tool_call. "
                    "You MUST respond ONLY with a JSON object in this exact format:\n"
                    '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n'
                    "No prose, no markdown, no explanation. JSON only."
                ),
            },
        ]

    raise RuntimeError(
        f"Ollama failed to produce a valid tool_call after {MAX_OLLAMA_RETRIES} attempts."
    )


# ------------------------------------------------------------------ #
# Anthropic                                                            #
# ------------------------------------------------------------------ #

def _anthropic_call(messages: list[dict]) -> dict:
    """
    Falls back to Anthropic claude-opus-4-8 with native tool-use API.
    Returns a tool_call dict: {"name": str, "arguments": dict}
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed") from exc

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Separate system prompt from conversation messages
    system_msg = ""
    conv_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            conv_messages.append(m)

    # Convert tool schemas to Anthropic format
    anthropic_tools = []
    for schema in _TOOL_SCHEMAS:
        anthropic_tools.append({
            "name": schema["name"],
            "description": schema.get("description", ""),
            "input_schema": schema.get("parameters", {"type": "object", "properties": {}}),
        })

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=system_msg or anthropic.NOT_GIVEN,
            tools=anthropic_tools or anthropic.NOT_GIVEN,
            messages=conv_messages,
            tool_choice={"type": "auto"},
        )
    except anthropic.APIError as exc:
        raise RuntimeError(f"Anthropic API error: {exc}") from exc

    _record_tokens(
        response.usage.input_tokens,
        response.usage.output_tokens,
        "anthropic",
    )

    # Extract tool use block
    for block in response.content:
        if block.type == "tool_use":
            return {"name": block.name, "arguments": block.input}

    # Model replied with text instead of a tool call — shouldn't happen with tool_choice=auto
    text_content = " ".join(
        b.text for b in response.content if hasattr(b, "text")
    )
    raise RuntimeError(
        f"Anthropic returned text instead of tool call: {text_content[:300]}"
    )


# ------------------------------------------------------------------ #
# Summarisation (for context window trimming)                          #
# ------------------------------------------------------------------ #

def summarise_history(old_messages: list[dict]) -> str:
    """
    Asks Ollama to produce a compact summary of old conversation messages.
    Used by the orchestrator to keep the context window under 40 messages.
    """
    prompt = (
        "Summarise the following conversation history between an autonomous CUDA kernel "
        "optimisation agent and its tools. Focus on: which kernel strategies were tried, "
        "which worked, which failed and why, and what the current best kernel looks like. "
        "Be concise — this summary replaces the original messages in the context window.\n\n"
        + json.dumps(old_messages, indent=2)
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        prompt_tokens = resp.json().get("prompt_eval_count", len(prompt) // 4)
        completion_tokens = resp.json().get("eval_count", len(content) // 4)
        _record_tokens(prompt_tokens, completion_tokens, "ollama")
        return content
    except Exception as exc:
        logger.warning("Summarisation failed: %s", exc)
        return "[Summary unavailable — history trimmed]"


# ------------------------------------------------------------------ #
# Public interface                                                      #
# ------------------------------------------------------------------ #

def call(messages: list[dict]) -> dict:
    """
    Main entry point. Returns tool_call dict: {"name": str, "arguments": dict}

    Routing:
      1. Try Ollama (up to MAX_OLLAMA_RETRIES)
      2. On failure, try Anthropic
      3. On failure, call human escalation callback and re-raise
    """
    # --- Ollama ---
    ollama_err_msg = ""
    try:
        return _ollama_call(messages)
    except RuntimeError as e:
        ollama_err_msg = str(e)
        logger.warning("Ollama exhausted, escalating to Anthropic: %s", e)

    # --- Anthropic ---
    try:
        return _anthropic_call(messages)
    except RuntimeError as anthropic_err:
        logger.error("Anthropic also failed: %s", anthropic_err)
        subject = "LLM routing failure"
        body = (
            f"Both Ollama and Anthropic failed to produce a valid tool call.\n\n"
            f"Ollama error: {ollama_err_msg}\n"
            f"Anthropic error: {anthropic_err}\n\n"
            "Reply with instructions for how to continue, or 'abort' to stop the run."
        )
        if _human_escalate:
            _human_escalate(subject, body)
        raise

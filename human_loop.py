"""
Telegram-based human-in-the-loop escalation.

Flow
----
1. send_alert(subject, body, state) — serialises state to disk FIRST (crash-safe),
   then sends a formatted Telegram message to TELEGRAM_CHAT_ID.
2. wait_for_reply(timeout_seconds) — long-polls the Telegram getUpdates API,
   blocking the caller until the human sends a reply.
3. The reply text is returned to the orchestrator and injected into the
   conversation history as a user message so the LLM sees it.

Required env vars
-----------------
  TELEGRAM_BOT_TOKEN   — from @BotFather
  TELEGRAM_CHAT_ID     — numeric chat/user ID (get via /start + getUpdates)
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from state import AgentState

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL = 3          # seconds between getUpdates polls
MAX_MESSAGE_LENGTH = 4096  # Telegram hard limit


class HumanLoop:
    def __init__(self) -> None:
        self._last_update_id: Optional[int] = None
        self._base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

    # ------------------------------------------------------------------ #
    # Send                                                                 #
    # ------------------------------------------------------------------ #

    def send_alert(
        self,
        subject: str,
        body: str,
        state: Optional["AgentState"] = None,
    ) -> None:
        """
        Write state to disk first, then send Telegram message.
        Splitting long messages to respect Telegram's 4096-char limit.
        """
        if state is not None:
            state.save()
            if state.human_interventions is not None:
                state.human_interventions += 1

        header = f"⚠️ *AGENT NEEDS INPUT*\n\n*{subject}*\n\n"
        footer = "\n\n_State saved to state.json — reply to resume._"
        full_text = header + body + footer

        # Split if over limit
        chunks = self._split_message(full_text)
        for chunk in chunks:
            self._send_message(chunk)

    def notify(self, message: str) -> None:
        """Non-blocking notification (no wait_for_reply). Used for budget alerts."""
        try:
            self._send_message(f"ℹ️ {message}")
        except Exception as exc:
            logger.warning("Non-blocking Telegram notify failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Receive                                                              #
    # ------------------------------------------------------------------ #

    def wait_for_reply(self, timeout_seconds: Optional[float] = None) -> str:
        """
        Block until the human sends a Telegram message.
        Returns the message text. If timeout_seconds elapses, returns
        "continue with best effort" and logs a warning.
        """
        logger.info("Waiting for human reply via Telegram...")
        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds else None

        # Drain any old updates so we don't pick up a stale message
        self._drain_updates()

        while True:
            if deadline and time.monotonic() > deadline:
                logger.warning("Human reply timeout — auto-resuming.")
                return "continue with best effort"

            updates = self._get_updates(timeout=POLL_INTERVAL)
            for update in updates:
                self._last_update_id = update["update_id"]
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue
                text = msg.get("text", "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if chat_id != TELEGRAM_CHAT_ID:
                    continue
                if not text:
                    continue
                logger.info("Human reply received: %.100s", text)
                self._send_message(f"✅ Got it. Resuming agent with your input.")
                return text

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _send_message(self, text: str) -> None:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured — printing alert to stdout:\n%s", text)
            print(f"\n[TELEGRAM ALERT]\n{text}\n")
            return
        url = f"{self._base}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            print(f"\n[TELEGRAM FALLBACK — send failed]\n{text}\n")

    def _get_updates(self, timeout: int = 3) -> list[dict]:
        params: dict = {"timeout": timeout, "allowed_updates": ["message"]}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1
        try:
            resp = requests.get(
                f"{self._base}/getUpdates",
                params=params,
                timeout=timeout + 5,
            )
            resp.raise_for_status()
            return resp.json().get("result", [])
        except requests.RequestException as exc:
            logger.warning("getUpdates failed: %s", exc)
            return []

    def _drain_updates(self) -> None:
        """Consume all pending updates so we start fresh."""
        updates = self._get_updates(timeout=1)
        if updates:
            self._last_update_id = updates[-1]["update_id"]

    @staticmethod
    def _split_message(text: str) -> list[str]:
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:MAX_MESSAGE_LENGTH])
            text = text[MAX_MESSAGE_LENGTH:]
        return chunks


# Module-level singleton used by orchestrator and token_budget
human_loop = HumanLoop()

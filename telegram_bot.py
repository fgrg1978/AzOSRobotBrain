# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Telegram bidirectional bot for the AzOS Robot Brain.

Polls the Bot API for incoming commands and forwards them to the brain.
Also exposes `notify(text, image)` for the brain to push alerts.

Commands:
  /help          — show available commands
  /status        — robot position, battery, mode
  /stop          — emergency stop
  /mode <name>   — switch operating mode
  /task <text>   — queue a free-text task (uses TaskPlanner)
  /photo         — request a camera snapshot

Usage:
    bot = TelegramBot(config["notifications"]["telegram"], brain)
    asyncio.create_task(bot.run())
"""

import asyncio
import json
import time
import urllib.parse
import urllib.request
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from server import BrainServer


# ── Telegram API helpers ──────────────────────────────────────────────────────


def _tg_request(bot_token: str, method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _tg_send_text(bot_token: str, chat_id: str, text: str) -> bool:
    r = _tg_request(
        bot_token, "sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    )
    return r.get("ok", False)


def _tg_send_photo(bot_token: str, chat_id: str, image: bytes, caption: str = "") -> bool:
    import io

    boundary = b"----TGBotBoundary"
    body = io.BytesIO()

    for name, value in [("chat_id", chat_id), ("caption", caption)]:
        body.write(b"--" + boundary + b"\r\n")
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(str(value).encode() + b"\r\n")

    body.write(b"--" + boundary + b"\r\n")
    body.write(b'Content-Disposition: form-data; name="photo"; filename="snap.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(image + b"\r\n")
    body.write(b"--" + boundary + b"--\r\n")

    raw = body.getvalue()
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    req.add_header("Content-Length", str(len(raw)))
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            r = json.loads(resp.read())
            return r.get("ok", False)
    except Exception:
        return False


def _tg_get_updates(bot_token: str, offset: int, timeout: int = 25) -> list[dict]:
    r = _tg_request(
        bot_token,
        "getUpdates",
        {"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
    )
    if r.get("ok"):
        return r.get("result", [])
    return []


# ── Bot ───────────────────────────────────────────────────────────────────────


class TelegramBot:
    """Long-poll Telegram bot connected to a BrainServer instance."""

    POLL_TIMEOUT = 25  # seconds — Telegram long-poll window

    def __init__(self, tg_config: dict, brain: "BrainServer"):
        # Token: prefer env var so the actual secret never lives in a
        # YAML file that might be world-readable or accidentally
        # committed. Falls back to config for backwards compatibility.
        # Chat-id is not secret — keep config support for it.
        import os as _os

        self.token = _os.environ.get("AZOS_BRAIN_TG_BOT_TOKEN", "") or tg_config.get(
            "bot_token", ""
        )
        self.chat_id = _os.environ.get("AZOS_BRAIN_TG_CHAT_ID", "") or tg_config.get("chat_id", "")
        self.brain = brain
        self._offset = 0
        self._running = False

    # ── Public notify ──────────────────────────────────────────────────────────

    async def notify(self, text: str, image: Optional[bytes] = None):
        """Push a message to the operator (called by brain on alerts)."""
        if not self.token or not self.chat_id:
            return
        if image:
            await asyncio.to_thread(_tg_send_photo, self.token, self.chat_id, image, text)
        else:
            await asyncio.to_thread(_tg_send_text, self.token, self.chat_id, text)

    # ── Polling loop ───────────────────────────────────────────────────────────

    async def run(self):
        """Start long-polling loop. Run as an asyncio task."""
        if not self.token:
            print("[TGBot] No bot_token — disabled")
            return
        self._running = True
        # Don't print full chat_id — it's not strictly secret, but a
        # public log shouldn't make it trivial to identify the operator.
        cid = str(self.chat_id)
        masked = (cid[:3] + "***" + cid[-2:]) if len(cid) >= 6 else "***"
        print(f"[TGBot] Polling (chat_id={masked})")
        while self._running:
            try:
                updates = await asyncio.to_thread(
                    _tg_get_updates, self.token, self._offset, self.POLL_TIMEOUT
                )
                for update in updates:
                    self._offset = update["update_id"] + 1
                    await self._handle(update)
            except Exception as e:
                print(f"[TGBot] Poll error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self._running = False

    # ── Command dispatcher ────────────────────────────────────────────────────

    async def _handle(self, update: dict):
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat = str(msg.get("chat", {}).get("id", ""))

        # Authorize: only our chat_id may send commands
        if chat != self.chat_id:
            print(f"[TGBot] Unauthorized message from chat_id={chat!r}")
            return

        if not text:
            return

        print(f"[TGBot] Received: {text!r}")

        if text.startswith("/help"):
            await self._reply(
                "Available commands:\n"
                "/status — robot state\n"
                "/stop — emergency stop\n"
                "/mode &lt;name&gt; — change mode\n"
                "/task &lt;description&gt; — queue task\n"
                "/photo — snapshot\n"
                "/help — this message"
            )

        elif text.startswith("/status"):
            await self._reply(self._format_status())

        elif text.startswith("/stop"):
            await self._reply("Sending emergency stop...")
            await self._do_stop()

        elif text.startswith("/mode"):
            parts = text.split(None, 1)
            if len(parts) < 2:
                await self._reply("Usage: /mode &lt;name&gt;  (e.g. /mode patrulla)")
            else:
                await self._do_mode(parts[1].strip())

        elif text.startswith("/task"):
            parts = text.split(None, 1)
            if len(parts) < 2:
                await self._reply("Usage: /task &lt;description&gt;")
            else:
                await self._do_task(parts[1].strip())

        elif text.startswith("/photo"):
            await self._do_photo()

        else:
            await self._reply("Unknown command. Send /help for the list.")

    # ── Command implementations ────────────────────────────────────────────────

    def _format_status(self) -> str:
        s = self.brain.state
        sensors = s.sensors
        odom = s.odom
        status = s.status
        lines = [
            f"<b>Robot Status</b>",
            f"Connected: {'yes' if s.connected else 'no'}",
            f"Mode: {self.brain.mode_manager.current_name if hasattr(self.brain, 'mode_manager') else 'n/a'}",
            f"Battery: {sensors.get('battery_mv', '?')} mV",
            f"Front range: {sensors.get('range_front_mm', '?')} mm",
            f"Right range: {sensors.get('range_right_mm', '?')} mm",
            f"Odom dist: {odom.get('dist_mm', 0)} mm",
            f"Heading: {odom.get('heading_cdeg', 0) / 100:.1f}°",
            f"Uptime: {status.get('uptime_s', '?')} s",
        ]
        return "\n".join(lines)

    async def _do_stop(self):
        # Robot-type-aware central stop (a drone must HOVER, not receive a
        # hardcoded diff-drive n_channels=2 zero frame) — see
        # BrainServer.emergency_stop(). runner.interrupt() is handled inside
        # emergency_stop() itself, so it isn't duplicated here.
        had_writer = getattr(self.brain, "_writer", None) is not None
        await self.brain.emergency_stop("Telegram /stop")
        if had_writer:
            await self._reply("Emergency stop sent.")
        else:
            await self._reply("Stop queued (no active connection).")

    async def _do_mode(self, name: str):
        if hasattr(self.brain, "mode_manager"):
            ok = self.brain.mode_manager.set_mode(name)
            if ok:
                await self._reply(f"Mode switched to <b>{name}</b>")
            else:
                modes = ", ".join(self.brain.mode_manager.modes.keys())
                await self._reply(f"Unknown mode '{name}'. Available: {modes}")
        else:
            await self._reply("Mode manager not available.")

    async def _do_task(self, description: str):
        if hasattr(self.brain, "task_queue"):
            await self.brain.task_queue.put(description)
            await self._reply(f"Task queued: <i>{description}</i>")
        else:
            await self._reply("Task queue not available.")

    async def _do_photo(self):
        image = self.brain.state.last_image
        if image:
            await self._reply_photo(image, "Latest camera frame")
        else:
            await self._reply("No image available yet.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _reply(self, text: str):
        await asyncio.to_thread(_tg_send_text, self.token, self.chat_id, text)

    async def _reply_photo(self, image: bytes, caption: str):
        await asyncio.to_thread(_tg_send_photo, self.token, self.chat_id, image, caption)

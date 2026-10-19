# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Notification backends: Pushover, Telegram, Email, Webhook.

All send methods are async (blocking I/O offloaded via asyncio.to_thread).
Uses only stdlib — no extra dependencies required.

Usage:
    notifier = Notifier(config["notifications"])
    await notifier.alert("Intruder detected", image_bytes=frame)
"""

import asyncio
import json
import smtplib
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass
class PushoverCfg:
    enabled: bool
    user_key: str
    api_token: str
    priority: int = 1
    sound: str = "siren"
    attach_image: bool = True


@dataclass
class TelegramCfg:
    enabled: bool
    bot_token: str
    chat_id: str
    commands: bool = True


@dataclass
class EmailCfg:
    enabled: bool
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    to: str


@dataclass
class WebhookCfg:
    enabled: bool
    url: str
    headers: dict = field(default_factory=dict)


@dataclass
class NotificationsCfg:
    pushover: PushoverCfg
    telegram: TelegramCfg
    email: EmailCfg
    webhook: WebhookCfg

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationsCfg":
        po = d.get("pushover", {})
        tg = d.get("telegram", {})
        em = d.get("email", {})
        wh = d.get("webhook", {})
        return cls(
            pushover=PushoverCfg(
                enabled=po.get("enabled", False),
                user_key=po.get("user_key", ""),
                api_token=po.get("api_token", ""),
                priority=po.get("priority", 1),
                sound=po.get("sound", "siren"),
                attach_image=po.get("attach_image", True),
            ),
            telegram=TelegramCfg(
                enabled=tg.get("enabled", False),
                bot_token=tg.get("bot_token", ""),
                chat_id=tg.get("chat_id", ""),
                commands=tg.get("commands", True),
            ),
            email=EmailCfg(
                enabled=em.get("enabled", False),
                smtp_host=em.get("smtp_host", "smtp.gmail.com"),
                smtp_port=em.get("smtp_port", 587),
                username=em.get("username", ""),
                password=em.get("password", ""),
                to=em.get("to", ""),
            ),
            webhook=WebhookCfg(
                enabled=wh.get("enabled", False),
                url=wh.get("url", ""),
                headers=wh.get("headers", {}),
            ),
        )


# ── HTTP helpers (sync, run in thread) ───────────────────────────────────────


def _http_post_json(url: str, payload: dict, headers: dict | None = None) -> int:
    """POST JSON, return HTTP status code."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def _http_post_multipart(url: str, fields: dict, files: dict | None = None) -> int:
    """POST multipart/form-data (for Pushover image attach, Telegram sendPhoto)."""
    import io, mimetypes

    boundary = b"----AzOSRobotBrainBoundary"
    body = io.BytesIO()

    for name, value in fields.items():
        body.write(b"--" + boundary + b"\r\n")
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(str(value).encode() + b"\r\n")

    for name, (filename, data, mime) in (files or {}).items():
        body.write(b"--" + boundary + b"\r\n")
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {mime}\r\n\r\n".encode())
        body.write(data + b"\r\n")

    body.write(b"--" + boundary + b"--\r\n")
    raw = body.getvalue()

    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    req.add_header("Content-Length", str(len(raw)))
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


# ── Backend implementations (sync) ────────────────────────────────────────────


def _pushover_send(
    cfg: PushoverCfg, title: str, message: str, image: Optional[bytes] = None
) -> bool:
    fields = {
        "token": cfg.api_token,
        "user": cfg.user_key,
        "title": title,
        "message": message,
        "priority": cfg.priority,
        "sound": cfg.sound,
    }
    if image and cfg.attach_image:
        status = _http_post_multipart(
            "https://api.pushover.net/1/messages.json",
            fields,
            files={"attachment": ("frame.jpg", image, "image/jpeg")},
        )
    else:
        status = _http_post_multipart("https://api.pushover.net/1/messages.json", fields)
    return status == 200


def _telegram_send_text(cfg: TelegramCfg, text: str) -> bool:
    url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
    status = _http_post_json(url, {"chat_id": cfg.chat_id, "text": text})
    return status == 200


def _telegram_send_photo(cfg: TelegramCfg, caption: str, image: bytes) -> bool:
    url = f"https://api.telegram.org/bot{cfg.bot_token}/sendPhoto"
    status = _http_post_multipart(
        url,
        {"chat_id": cfg.chat_id, "caption": caption},
        files={"photo": ("frame.jpg", image, "image/jpeg")},
    )
    return status == 200


def _email_send(cfg: EmailCfg, subject: str, body: str, image: Optional[bytes] = None) -> bool:
    msg: MIMEMultipart | MIMEText
    if image:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "plain"))
        img_part = MIMEImage(image, _subtype="jpeg")
        img_part.add_header("Content-Disposition", "attachment", filename="frame.jpg")
        msg.attach(img_part)
    else:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "plain"))

    msg["Subject"] = subject
    msg["From"] = cfg.username
    msg["To"] = cfg.to

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(cfg.username, cfg.password)
            server.sendmail(cfg.username, cfg.to, msg.as_string())
        return True
    except Exception:
        return False


def _webhook_send(cfg: WebhookCfg, title: str, message: str) -> bool:
    payload = {"title": title, "message": message, "source": "AzOSRobotBrain"}
    status = _http_post_json(cfg.url, payload, cfg.headers)
    return status in (200, 201, 202, 204)


# ── Notifier ─────────────────────────────────────────────────────────────────


class Notifier:
    """Async notification dispatcher. Fires all enabled backends in parallel."""

    def __init__(self, config: dict):
        self.cfg = NotificationsCfg.from_dict(config)

    async def alert(
        self, message: str, title: str = "Robot Alert", image: Optional[bytes] = None
    ) -> dict[str, bool]:
        """Send alert through all enabled backends.

        Returns dict of backend -> success.
        """
        tasks = {}

        if self.cfg.pushover.enabled and self.cfg.pushover.api_token:
            tasks["pushover"] = asyncio.to_thread(
                _pushover_send, self.cfg.pushover, title, message, image
            )

        if self.cfg.telegram.enabled and self.cfg.telegram.bot_token:
            if image:
                tasks["telegram"] = asyncio.to_thread(
                    _telegram_send_photo, self.cfg.telegram, f"{title}: {message}", image
                )
            else:
                tasks["telegram"] = asyncio.to_thread(
                    _telegram_send_text, self.cfg.telegram, f"{title}: {message}"
                )

        if self.cfg.email.enabled and self.cfg.email.username:
            tasks["email"] = asyncio.to_thread(_email_send, self.cfg.email, title, message, image)

        if self.cfg.webhook.enabled and self.cfg.webhook.url:
            tasks["webhook"] = asyncio.to_thread(_webhook_send, self.cfg.webhook, title, message)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {name: (r is True) for name, r in zip(tasks.keys(), results)}

    async def info(self, message: str) -> dict[str, bool]:
        """Send low-priority info (Telegram only, no Pushover siren)."""
        tasks = {}
        if self.cfg.telegram.enabled and self.cfg.telegram.bot_token:
            tasks["telegram"] = asyncio.to_thread(
                _telegram_send_text, self.cfg.telegram, f"[INFO] {message}"
            )
        if self.cfg.webhook.enabled and self.cfg.webhook.url:
            tasks["webhook"] = asyncio.to_thread(
                _webhook_send, self.cfg.webhook, "Robot Info", message
            )
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {name: (r is True) for name, r in zip(tasks.keys(), results)}

"""Email channel must NOT send progress / tool-hint / retry-wait messages.

Regression for the bug surfaced 2026-05-05: agent.loop publishes one
OutboundMessage per tool-call iteration with metadata._progress so that
streaming UIs (askola web) can render a live trace. Email is delivery-
grade, so sending each iteration produces a 30-40x spam amplification.
The fix: filter at email channel send().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.email import EmailChannel, EmailConfig


def _make_channel():
    cfg = EmailConfig(
        enabled=True,
        consent_granted=True,
        imap_host="imap.example.com",
        imap_username="bot@example.com",
        imap_password="x",
        smtp_host="smtp.example.com",
        smtp_username="bot@example.com",
        smtp_password="x",
        from_address="bot@example.com",
    )
    bus = MessageBus()
    return EmailChannel(cfg, bus)


@pytest.mark.asyncio
async def test_progress_message_is_dropped():
    ch = _make_channel()
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="step 12: reading file...",
            metadata={"_progress": True},
        ))
    smtp_mock.assert_not_called()


@pytest.mark.asyncio
async def test_tool_hint_message_is_dropped():
    ch = _make_channel()
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="hint",
            metadata={"_tool_hint": True},
        ))
    smtp_mock.assert_not_called()


@pytest.mark.asyncio
async def test_retry_wait_message_is_dropped():
    ch = _make_channel()
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="retry...",
            metadata={"_retry_wait": True},
        ))
    smtp_mock.assert_not_called()


@pytest.mark.asyncio
async def test_normal_message_passes_through():
    ch = _make_channel()
    # Pre-populate _last_subject_by_chat to simulate prior inbound (so this
    # is a "reply" scenario; auto_reply_enabled=true by default lets it pass).
    ch._last_subject_by_chat["user@example.com"] = "Original subject"
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="Hello, here is your quote",
            metadata={},
        ))
    smtp_mock.assert_called_once()


@pytest.mark.asyncio
async def test_message_with_no_metadata_passes_through():
    ch = _make_channel()
    ch._last_subject_by_chat["user@example.com"] = "Original subject"
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="Hello",
            metadata=None,
        ))
    smtp_mock.assert_called_once()


@pytest.mark.asyncio
async def test_progress_with_extra_metadata_keys_still_dropped():
    """Defensive: even if other metadata keys present, _progress dominates."""
    ch = _make_channel()
    with patch.object(ch, "_smtp_send", new=AsyncMock()) as smtp_mock:
        await ch.send(OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="step",
            metadata={"_progress": True, "subject": "should not be used"},
        ))
    smtp_mock.assert_not_called()

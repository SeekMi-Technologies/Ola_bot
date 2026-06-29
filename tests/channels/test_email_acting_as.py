"""Email channel sender pre-lookup → acting-as resolution.

Channel resolves sender email → admin._id via MCP salesperson.lookup_by_email
BEFORE publishing inbound. The result rides on InboundMessage.metadata
["_acting_as"] because asyncio ContextVar does not propagate across the bus
queue (channel polling task → agent.loop.run task → _dispatch new task).

Unknown senders are SMTP-rejected and never reach the agent. Transport / token
errors fail closed: drop the email, no reply, no agent dispatch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.email import EmailChannel, EmailConfig


async def _run_one_polling_cycle(ch: EmailChannel) -> None:
    """Start the polling loop, give one cycle to run, then cancel.

    The loop's `await asyncio.sleep(poll_seconds)` floors at 5s so flipping
    `_running` cannot wake it; cancellation is the only fast exit.
    """
    task = asyncio.create_task(ch.start())
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)


def _make_channel(**overrides):
    cfg_kwargs = dict(
        enabled=True,
        consent_granted=True,
        imap_host="imap.example.com",
        imap_username="bot@example.com",
        imap_password="x",
        smtp_host="smtp.example.com",
        smtp_username="bot@example.com",
        smtp_password="x",
        from_address="bot@example.com",
        allow_from=["*"],
    )
    cfg_kwargs.update(overrides)
    cfg = EmailConfig(**cfg_kwargs)
    bus = MessageBus()
    return EmailChannel(cfg, bus)


def _envelope(found: bool, admin_id: str | None = None) -> str:
    if found:
        return json.dumps(
            {
                "ok": True,
                "data": {
                    "found": True,
                    "salesperson": {
                        "_id": admin_id,
                        "email": "yz@example.com",
                        "name": "Yz",
                        "surname": "Sales",
                        "role": "admin",
                        "language": "en_us",
                    },
                },
            }
        )
    return json.dumps(
        {"ok": True, "data": {"found": False, "message": "No matching salesperson"}}
    )


def _mock_call_tool_result(text: str, is_error: bool = False):
    text_block = MagicMock()
    text_block.text = text
    result = MagicMock()
    result.isError = is_error
    result.content = [text_block]
    return result


class _FakeSession:
    def __init__(self, call_tool_result):
        self._result = call_tool_result
        self.initialize = AsyncMock()
        self.call_tool = AsyncMock(return_value=call_tool_result)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeStreamableClient:
    def __init__(self):
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return (MagicMock(), MagicMock(), MagicMock())

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def mcp_token(monkeypatch):
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "test-token")


@pytest.mark.asyncio
async def test_known_sender_resolves_to_admin_id(mcp_token):
    ch = _make_channel()
    fake_result = _mock_call_tool_result(_envelope(True, "67890abcdef0123456789abc"))
    fake_session = _FakeSession(fake_result)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=_FakeStreamableClient(),
        ),
        patch("mcp.ClientSession", return_value=fake_session),
    ):
        admin_id = await ch._resolve_sender_acting_as("yz@example.com")

    assert admin_id == "67890abcdef0123456789abc"
    fake_session.initialize.assert_awaited_once()
    fake_session.call_tool.assert_awaited_once_with(
        "salesperson.lookup_by_email", {"email": "yz@example.com"}
    )


@pytest.mark.asyncio
async def test_unknown_sender_returns_none(mcp_token):
    ch = _make_channel()
    fake_result = _mock_call_tool_result(_envelope(False))
    fake_session = _FakeSession(fake_result)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=_FakeStreamableClient(),
        ),
        patch("mcp.ClientSession", return_value=fake_session),
    ):
        admin_id = await ch._resolve_sender_acting_as("stranger@example.com")

    assert admin_id is None


@pytest.mark.asyncio
async def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    ch = _make_channel()

    with pytest.raises(RuntimeError, match="MCP_SERVICE_TOKEN"):
        await ch._resolve_sender_acting_as("anyone@example.com")


@pytest.mark.asyncio
async def test_mcp_iserror_raises(mcp_token):
    ch = _make_channel()
    fake_result = _mock_call_tool_result(
        json.dumps({"ok": False, "code": "VALIDATION", "message": "bad email"}),
        is_error=True,
    )
    fake_session = _FakeSession(fake_result)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=_FakeStreamableClient(),
        ),
        patch("mcp.ClientSession", return_value=fake_session),
    ):
        with pytest.raises(RuntimeError, match="isError"):
            await ch._resolve_sender_acting_as("bad@example.com")


@pytest.mark.asyncio
async def test_polling_loop_known_sender_publishes_with_acting_as(mcp_token):
    """One-cycle polling: known sender → metadata['_acting_as'] set → bus."""
    ch = _make_channel()
    item = {
        "sender": "yz@example.com",
        "subject": "Inquiry",
        "message_id": "<id-1@example.com>",
        "content": "[EMAIL-CONTEXT]...",
        "metadata": {"message_id": "<id-1@example.com>", "subject": "Inquiry"},
        "media": [],
    }

    with (
        patch.object(ch, "_fetch_new_messages", return_value=[item]),
        patch.object(
            ch, "_resolve_sender_acting_as", new=AsyncMock(return_value="adminA")
        ),
        patch.object(ch, "_send_unknown_sender_reply", new=AsyncMock()) as reject,
        patch.object(ch, "_handle_message", new=AsyncMock()) as handle,
    ):
        await _run_one_polling_cycle(ch)

    handle.assert_awaited_once()
    kwargs = handle.await_args.kwargs
    assert kwargs["sender_id"] == "yz@example.com"
    assert kwargs["metadata"]["_acting_as"] == "adminA"
    # Existing metadata fields preserved.
    assert kwargs["metadata"]["subject"] == "Inquiry"
    reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_polling_loop_unknown_sender_replies_and_does_not_publish(mcp_token):
    ch = _make_channel()
    item = {
        "sender": "stranger@example.com",
        "subject": "Hi",
        "message_id": "<id-2@example.com>",
        "content": "...",
        "metadata": {"message_id": "<id-2@example.com>"},
        "media": [],
    }

    with (
        patch.object(ch, "_fetch_new_messages", return_value=[item]),
        patch.object(
            ch, "_resolve_sender_acting_as", new=AsyncMock(return_value=None)
        ),
        patch.object(ch, "_send_unknown_sender_reply", new=AsyncMock()) as reject,
        patch.object(ch, "_handle_message", new=AsyncMock()) as handle,
    ):
        await _run_one_polling_cycle(ch)

    reject.assert_awaited_once_with("stranger@example.com")
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_polling_loop_lookup_error_drops_email(mcp_token):
    """Transport error → log + drop. No SMTP reply, no bus publish."""
    ch = _make_channel()
    item = {
        "sender": "x@example.com",
        "subject": "S",
        "message_id": "<id-3@example.com>",
        "content": "...",
        "metadata": {},
        "media": [],
    }

    with (
        patch.object(ch, "_fetch_new_messages", return_value=[item]),
        patch.object(
            ch,
            "_resolve_sender_acting_as",
            new=AsyncMock(side_effect=RuntimeError("MCP unreachable")),
        ),
        patch.object(ch, "_send_unknown_sender_reply", new=AsyncMock()) as reject,
        patch.object(ch, "_handle_message", new=AsyncMock()) as handle,
    ):
        await _run_one_polling_cycle(ch)

    reject.assert_not_awaited()
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_inbound_each_carries_own_acting_as(mcp_token):
    """Two senders in one polling cycle each get their own _acting_as."""
    ch = _make_channel()
    items = [
        {
            "sender": "a@example.com",
            "subject": "S1",
            "message_id": "<a@x>",
            "content": "...",
            "metadata": {},
            "media": [],
        },
        {
            "sender": "b@example.com",
            "subject": "S2",
            "message_id": "<b@x>",
            "content": "...",
            "metadata": {},
            "media": [],
        },
    ]
    resolutions = {"a@example.com": "adminA", "b@example.com": "adminB"}

    async def _fake_resolve(sender):
        return resolutions[sender]

    with (
        patch.object(ch, "_fetch_new_messages", return_value=items),
        patch.object(ch, "_resolve_sender_acting_as", new=AsyncMock(side_effect=_fake_resolve)),
        patch.object(ch, "_handle_message", new=AsyncMock()) as handle,
    ):
        await _run_one_polling_cycle(ch)

    assert handle.await_count == 2
    seen = {
        call.kwargs["sender_id"]: call.kwargs["metadata"]["_acting_as"]
        for call in handle.await_args_list
    }
    assert seen == {"a@example.com": "adminA", "b@example.com": "adminB"}


@pytest.mark.asyncio
async def test_dispatch_propagates_acting_as_to_contextvar(tmp_path):
    """agent.loop._dispatch must call set_acting_as with metadata['_acting_as']."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage

    msg = InboundMessage(
        channel="email",
        sender_id="x@example.com",
        chat_id="x@example.com",
        content="...",
        media=[],
        metadata={"_acting_as": "adminZ"},
    )

    set_calls: list[str | None] = []

    def _capture(value):
        set_calls.append(value)

    with patch("nanobot.agent.loop.set_acting_as", side_effect=_capture):
        # Stop _dispatch before it runs the heavy LLM path: we only need the
        # first line (set_acting_as call) to fire. Force _effective_session_key
        # to raise so we exit early after _connect_mcp early-returns.
        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = tmp_path  # _dispatch provisions the acting admin (#354)
        loop._mcp_connected = True  # Skip _connect_mcp work
        loop._mcp_connecting = False
        loop._mcp_servers = {}
        loop._effective_session_key = MagicMock(side_effect=RuntimeError("stop"))
        with pytest.raises(RuntimeError, match="stop"):
            await loop._dispatch(msg)

    assert set_calls == ["adminZ"]

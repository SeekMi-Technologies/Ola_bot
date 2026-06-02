"""Tests for WhatsApp channel outbound media support."""

import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.channels.whatsapp import (
    WhatsAppChannel,
    _load_or_create_bridge_token,
)


def _make_channel() -> WhatsAppChannel:
    bus = MagicMock()
    ch = WhatsAppChannel({"enabled": True}, bus)
    ch._ws = AsyncMock()
    ch._connected = True
    return ch


@pytest.mark.asyncio
async def test_send_text_only():
    ch = _make_channel()
    msg = OutboundMessage(channel="whatsapp", chat_id="123@s.whatsapp.net", content="hello")

    await ch.send(msg)

    ch._ws.send.assert_called_once()
    payload = json.loads(ch._ws.send.call_args[0][0])
    assert payload["type"] == "send"
    assert payload["text"] == "hello"


@pytest.mark.asyncio
async def test_send_media_dispatches_send_media_command():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="check this out",
        media=["/tmp/photo.jpg"],
    )

    await ch.send(msg)

    assert ch._ws.send.call_count == 2
    text_payload = json.loads(ch._ws.send.call_args_list[0][0][0])
    media_payload = json.loads(ch._ws.send.call_args_list[1][0][0])

    assert text_payload["type"] == "send"
    assert text_payload["text"] == "check this out"

    assert media_payload["type"] == "send_media"
    assert media_payload["filePath"] == "/tmp/photo.jpg"
    assert media_payload["mimetype"] == "image/jpeg"
    assert media_payload["fileName"] == "photo.jpg"


@pytest.mark.asyncio
async def test_send_media_only_no_text():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="",
        media=["/tmp/doc.pdf"],
    )

    await ch.send(msg)

    ch._ws.send.assert_called_once()
    payload = json.loads(ch._ws.send.call_args[0][0])
    assert payload["type"] == "send_media"
    assert payload["mimetype"] == "application/pdf"


@pytest.mark.asyncio
async def test_send_multiple_media():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="",
        media=["/tmp/a.png", "/tmp/b.mp4"],
    )

    await ch.send(msg)

    assert ch._ws.send.call_count == 2
    p1 = json.loads(ch._ws.send.call_args_list[0][0][0])
    p2 = json.loads(ch._ws.send.call_args_list[1][0][0])
    assert p1["mimetype"] == "image/png"
    assert p2["mimetype"] == "video/mp4"


@pytest.mark.asyncio
async def test_send_when_disconnected_is_noop():
    ch = _make_channel()
    ch._connected = False

    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="hello",
        media=["/tmp/x.jpg"],
    )
    await ch.send(msg)

    ch._ws.send.assert_not_called()


@pytest.mark.asyncio
async def test_group_policy_mention_skips_unmentioned_group_message():
    ch = WhatsAppChannel({"enabled": True, "groupPolicy": "mention"}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps(
            {
                "type": "message",
                "id": "m1",
                "sender": "12345@g.us",
                "pn": "user@s.whatsapp.net",
                "content": "hello group",
                "timestamp": 1,
                "isGroup": True,
                "wasMentioned": False,
            }
        )
    )

    ch._handle_message.assert_not_called()


@pytest.mark.skip(
    reason="P0 multi-tenant: all group messages are force-dropped (Baileys group "
    "bugs #1505/#1935/#2233); group_policy 'mention' / whitelist re-enable → handoff."
)
@pytest.mark.asyncio
async def test_group_policy_mention_accepts_mentioned_group_message():
    ch = WhatsAppChannel({"enabled": True, "groupPolicy": "mention"}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps(
            {
                "type": "message",
                "id": "m1",
                "sender": "12345@g.us",
                "pn": "user@s.whatsapp.net",
                "content": "hello @bot",
                "timestamp": 1,
                "isGroup": True,
                "wasMentioned": True,
            }
        )
    )

    ch._handle_message.assert_awaited_once()
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["chat_id"] == "12345@g.us"
    assert kwargs["sender_id"] == "user"


# ---------------------------------------------------------------------------
# Multi-tenant tests (admin_id set → HMAC token + _acting_as injection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acting_as_injected_when_admin_id_set():
    """metadata._acting_as = self._admin_id when admin_id is set (★ 隔离命门)."""
    admin_id = "507f1f77bcf86cd799439011"
    ch = WhatsAppChannel({"enabled": True, "adminId": admin_id}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "mt1",
            "sender": "12345@s.whatsapp.net",
            "pn": "",
            "content": "hello",
            "timestamp": 1,
        })
    )

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["metadata"]["_acting_as"] == admin_id


@pytest.mark.asyncio
async def test_acting_as_not_injected_when_admin_id_empty():
    """Legacy single-tenant mode (admin_id='') does NOT inject _acting_as."""
    ch = WhatsAppChannel({"enabled": True}, MagicMock())  # admin_id defaults to ""
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "lg1",
            "sender": "12345@s.whatsapp.net",
            "pn": "",
            "content": "hello",
            "timestamp": 1,
        })
    )

    kwargs = ch._handle_message.await_args.kwargs
    assert "_acting_as" not in kwargs["metadata"]


@pytest.mark.asyncio
async def test_group_messages_always_dropped_p0():
    """P0: all group messages dropped regardless of group_policy (Baileys group bugs)."""
    ch = WhatsAppChannel({"enabled": True, "adminId": "507f1f77bcf86cd799439011"}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "g1",
            "sender": "12345@g.us",
            "pn": "user@s.whatsapp.net",
            "content": "hello @bot",
            "timestamp": 1,
            "isGroup": True,
            "wasMentioned": True,  # even mentioned → still dropped in P0
        })
    )

    ch._handle_message.assert_not_called()


def test_effective_bridge_token_uses_hmac_when_admin_id_set(monkeypatch):
    """Multi-tenant token = HMAC-SHA256(MCP_SERVICE_TOKEN, admin_id).hex().

    Same algorithm as bridge/src/server.ts tokenFor() — verifies cross-language
    byte equality so bridge + nanobot derive identical tokens from the shared env.
    """
    import hashlib
    import hmac as _hmac

    admin_id = "507f1f77bcf86cd799439011"
    secret = "TEST_SECRET_123"
    monkeypatch.setenv("MCP_SERVICE_TOKEN", secret)

    ch = WhatsAppChannel({"enabled": True, "adminId": admin_id}, MagicMock())
    token = ch._effective_bridge_token()

    expected = _hmac.new(secret.encode(), admin_id.encode(), hashlib.sha256).hexdigest()
    assert token == expected
    # Sanity: pre-computed value from `node -e "..."` cross-check (run in shell)
    assert token == "d2a42894f58022dadfbce3065430fa4f8791e2f14470fcb011939981526e5114"


def test_effective_bridge_token_raises_when_admin_id_set_but_secret_missing(monkeypatch):
    """Multi-tenant mode requires MCP_SERVICE_TOKEN env."""
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    ch = WhatsAppChannel(
        {"enabled": True, "adminId": "507f1f77bcf86cd799439011"}, MagicMock()
    )

    with pytest.raises(RuntimeError, match="MCP_SERVICE_TOKEN"):
        ch._effective_bridge_token()


@pytest.mark.asyncio
async def test_start_uses_multi_tenant_url_when_admin_id_set(monkeypatch, tmp_path):
    """Multi-tenant mode: ws_url = <base>/wa/<adminId>?token=<hmac>; no auth msg sent."""
    admin_id = "507f1f77bcf86cd799439011"
    secret = "TEST_SECRET_123"
    monkeypatch.setenv("MCP_SERVICE_TOKEN", secret)
    # Isolate from real ~/.nanobot/wa/bridge.port so _resolve_ws_url falls back to config.bridge_url
    monkeypatch.setattr("nanobot.channels.whatsapp.Path.home", lambda: tmp_path)

    captured_url: list[str] = []
    sent_messages: list[str] = []

    class FakeWS:
        def __init__(self):
            self.close = AsyncMock()

        async def send(self, m):
            sent_messages.append(m)

        def __aiter__(self):
            # Break the outer reconnect loop after first iteration (multi-tenant
            # mode doesn't send an auth message, so we can't hook via send())
            ch._running = False
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeConnect:
        def __init__(self, url):
            captured_url.append(url)
            self.ws = FakeWS()

        async def __aenter__(self):
            return self.ws

        async def __aexit__(self, *a):
            return False

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda url: FakeConnect(url)),
    )

    ch = WhatsAppChannel(
        {"enabled": True, "adminId": admin_id, "bridgeUrl": "ws://127.0.0.1:54321"},
        MagicMock(),
    )
    await ch.start()

    # URL should be /wa/<adminId>?token=<hmac>
    assert len(captured_url) == 1
    assert captured_url[0].startswith(f"ws://127.0.0.1:54321/wa/{admin_id}?token=")
    # Multi-tenant: no auth message (token verified by bridge during WS upgrade)
    assert sent_messages == []


@pytest.mark.asyncio
async def test_sender_id_prefers_phone_jid_over_lid():
    """sender_id should resolve to phone number when @s.whatsapp.net JID is present."""
    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "lid1",
            "sender": "ABC123@lid.whatsapp.net",
            "pn": "5551234@s.whatsapp.net",
            "content": "hi",
            "timestamp": 1,
        })
    )

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "5551234"


@pytest.mark.asyncio
async def test_lid_to_phone_cache_resolves_lid_only_messages():
    """When only LID is present, a cached LID→phone mapping should be used."""
    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    ch._handle_message = AsyncMock()

    # First message: both phone and LID → builds cache
    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "c1",
            "sender": "LID99@lid.whatsapp.net",
            "pn": "5559999@s.whatsapp.net",
            "content": "first",
            "timestamp": 1,
        })
    )
    # Second message: only LID, no phone
    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "c2",
            "sender": "LID99@lid.whatsapp.net",
            "pn": "",
            "content": "second",
            "timestamp": 2,
        })
    )

    second_kwargs = ch._handle_message.await_args_list[1].kwargs
    assert second_kwargs["sender_id"] == "5559999"


@pytest.mark.asyncio
async def test_voice_message_transcription_uses_media_path():
    """Voice messages are transcribed when media path is available."""
    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    ch.transcription_provider = "openai"
    ch.transcription_api_key = "sk-test"
    ch._handle_message = AsyncMock()
    ch.transcribe_audio = AsyncMock(return_value="Hello world")

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "v1",
            "sender": "12345@s.whatsapp.net",
            "pn": "",
            "content": "[Voice Message]",
            "timestamp": 1,
            "media": ["/tmp/voice.ogg"],
        })
    )

    ch.transcribe_audio.assert_awaited_once_with("/tmp/voice.ogg")
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["content"].startswith("Hello world")


@pytest.mark.asyncio
async def test_voice_message_no_media_shows_not_available():
    """Voice messages without media produce a fallback placeholder."""
    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    ch._handle_message = AsyncMock()

    await ch._handle_bridge_message(
        json.dumps({
            "type": "message",
            "id": "v2",
            "sender": "12345@s.whatsapp.net",
            "pn": "",
            "content": "[Voice Message]",
            "timestamp": 1,
        })
    )

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["content"] == "[Voice Message: Audio not available]"


def test_load_or_create_bridge_token_persists_generated_secret(tmp_path):
    token_path = tmp_path / "whatsapp-auth" / "bridge-token"

    first = _load_or_create_bridge_token(token_path)
    second = _load_or_create_bridge_token(token_path)

    assert first == second
    assert token_path.read_text(encoding="utf-8") == first
    assert len(first) >= 32
    if os.name != "nt":
        assert token_path.stat().st_mode & 0o777 == 0o600


def test_configured_bridge_token_skips_local_token_file(monkeypatch, tmp_path):
    token_path = tmp_path / "whatsapp-auth" / "bridge-token"
    monkeypatch.setattr("nanobot.channels.whatsapp._bridge_token_path", lambda: token_path)
    ch = WhatsAppChannel({"enabled": True, "bridgeToken": "manual-secret"}, MagicMock())

    assert ch._effective_bridge_token() == "manual-secret"
    assert not token_path.exists()


@pytest.mark.asyncio
async def test_login_exports_canonical_env_vars(monkeypatch, tmp_path):
    """login() spawns the multi-tenant bridge — must export MCP_SERVICE_TOKEN
    + AUTH_ROOT (the new env contract), not the legacy BRIDGE_TOKEN/AUTH_DIR."""
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "TEST_SECRET_123")
    calls = []

    monkeypatch.setattr("nanobot.channels.whatsapp._ensure_bridge_setup", lambda: bridge_dir)
    monkeypatch.setattr("nanobot.channels.whatsapp.shutil.which", lambda _: "/usr/bin/npm")

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return MagicMock()

    monkeypatch.setattr("nanobot.channels.whatsapp.subprocess.run", fake_run)
    admin_id = "507f1f77bcf86cd799439011"
    ch = WhatsAppChannel({"enabled": True, "adminId": admin_id}, MagicMock())

    assert await ch.login() is True
    assert len(calls) == 1

    _, kwargs = calls[0]
    assert kwargs["cwd"] == bridge_dir
    assert kwargs["env"]["MCP_SERVICE_TOKEN"] == "TEST_SECRET_123"
    assert "AUTH_ROOT" in kwargs["env"]
    assert ".nanobot/wa" in kwargs["env"]["AUTH_ROOT"]
    # When admin_id set, login also restricts the spawned bridge to that admin
    assert kwargs["env"]["SINGLE_ADMIN_ID"] == admin_id
    # Legacy env vars (replaced) should NOT be set
    assert "BRIDGE_TOKEN" not in kwargs["env"] or kwargs["env"].get("BRIDGE_TOKEN") == ""
    assert "AUTH_DIR" not in kwargs["env"] or "wa" in kwargs["env"].get("AUTH_DIR", "")


@pytest.mark.asyncio
async def test_login_fails_when_mcp_service_token_missing(monkeypatch, tmp_path):
    """login() must hard-fail (return False) when MCP_SERVICE_TOKEN is unset —
    spawning the bridge without it produces an instant exit-1 + confusing error."""
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr("nanobot.channels.whatsapp._ensure_bridge_setup", lambda: bridge_dir)

    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    assert await ch.login() is False


@pytest.mark.asyncio
async def test_start_sends_auth_message_with_generated_token(monkeypatch, tmp_path):
    token_path = tmp_path / "whatsapp-auth" / "bridge-token"
    sent_messages: list[str] = []

    class FakeWS:
        def __init__(self) -> None:
            self.close = AsyncMock()

        async def send(self, message: str) -> None:
            sent_messages.append(message)
            ch._running = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeConnect:
        def __init__(self, ws):
            self.ws = ws

        async def __aenter__(self):
            return self.ws

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("nanobot.channels.whatsapp._bridge_token_path", lambda: token_path)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda url: FakeConnect(FakeWS())),
    )

    ch = WhatsAppChannel({"enabled": True, "bridgeUrl": "ws://localhost:3001"}, MagicMock())
    await ch.start()

    assert sent_messages == [
        json.dumps({"type": "auth", "token": token_path.read_text(encoding="utf-8")})
    ]

"""Tests for WhatsApp multi-tenant registry (file-system discovery + poll loop).

Coverage:
- _scan_admin_dirs: ObjectId hex filter, missing auth/ subdir guard
- _read_bridge_port: invalid/missing/zero handling
- WhatsAppMultiTenantRegistry.expand: per-admin instantiation, transcription
  propagation, no-portfile graceful fallback, no-channel inert mode
- WhatsAppMultiTenantRegistry.tick: add new admin, remove disappeared admin
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.channels import whatsapp_registry as wr
from nanobot.channels.whatsapp import WhatsAppChannel


@pytest.fixture
def tmp_wa_root(tmp_path, monkeypatch):
    """Redirect WA_ROOT + PORTFILE to a per-test tmp directory."""
    wa_root = tmp_path / "wa"
    wa_root.mkdir()
    monkeypatch.setattr(wr, "WA_ROOT", wa_root)
    monkeypatch.setattr(wr, "PORTFILE", wa_root / "bridge.port")
    return wa_root


def _make_admin_dir(wa_root: Path, admin_id: str) -> None:
    (wa_root / admin_id / "auth").mkdir(parents=True)


def _write_portfile(wa_root: Path, port: int) -> None:
    (wa_root / "bridge.port").write_text(str(port))


def _make_manager_with_placeholder() -> MagicMock:
    """Fake ChannelManager with a single 'whatsapp' placeholder channel."""
    manager = MagicMock()
    manager.bus = MagicMock()
    placeholder = WhatsAppChannel({"enabled": True}, manager.bus)
    placeholder.transcription_provider = "openai"
    placeholder.transcription_api_key = "sk-test"
    placeholder.transcription_api_base = ""
    placeholder.transcription_language = "zh"
    manager.channels = {"whatsapp": placeholder}
    return manager


# ---------------------------------------------------------------------------
# _scan_admin_dirs
# ---------------------------------------------------------------------------


def test_scan_admin_dirs_finds_valid_object_ids(tmp_wa_root):
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    _make_admin_dir(tmp_wa_root, "6a06c9c58e6056b24664de25")

    assert wr._scan_admin_dirs() == {
        "507f1f77bcf86cd799439011",
        "6a06c9c58e6056b24664de25",
    }


def test_scan_admin_dirs_skips_non_object_id_names(tmp_wa_root):
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")  # valid
    _make_admin_dir(tmp_wa_root, "not-an-objectid")           # too short + non-hex
    _make_admin_dir(tmp_wa_root, "ZZZZZZZZZZZZZZZZZZZZZZZZ")  # len 24 but uppercase

    assert wr._scan_admin_dirs() == {"507f1f77bcf86cd799439011"}


def test_scan_admin_dirs_returns_empty_when_no_dirs(tmp_wa_root):
    assert wr._scan_admin_dirs() == set()


def test_scan_admin_dirs_skips_dir_without_auth_subdir(tmp_wa_root):
    (tmp_wa_root / "507f1f77bcf86cd799439011").mkdir()  # no /auth subdir
    assert wr._scan_admin_dirs() == set()


# ---------------------------------------------------------------------------
# _read_bridge_port
# ---------------------------------------------------------------------------


def test_read_bridge_port_returns_port(tmp_wa_root):
    _write_portfile(tmp_wa_root, 54321)
    assert wr._read_bridge_port() == 54321


def test_read_bridge_port_returns_none_when_missing(tmp_wa_root):
    assert wr._read_bridge_port() is None


def test_read_bridge_port_returns_none_on_invalid_content(tmp_wa_root):
    (tmp_wa_root / "bridge.port").write_text("not-a-number")
    assert wr._read_bridge_port() is None


def test_read_bridge_port_returns_none_on_zero(tmp_wa_root):
    _write_portfile(tmp_wa_root, 0)
    assert wr._read_bridge_port() is None


# ---------------------------------------------------------------------------
# WhatsAppMultiTenantRegistry.expand
# ---------------------------------------------------------------------------


def test_registry_inert_when_no_whatsapp_channel(tmp_wa_root):
    manager = MagicMock()
    manager.channels = {}
    registry = wr.WhatsAppMultiTenantRegistry(manager)

    assert registry.base_config is None
    registry.expand()  # no-op
    assert manager.channels == {}


def test_registry_expand_creates_per_admin_channels(tmp_wa_root):
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    _make_admin_dir(tmp_wa_root, "6a06c9c58e6056b24664de25")
    # Portfile no longer required at expand time — channels resolve port dynamically
    # per reconnect (supports both shared-bridge and multi-bridge modes).

    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()

    # Placeholder removed; 2 per-admin instances added
    assert "whatsapp" not in manager.channels
    assert "whatsapp:507f1f77bcf86cd799439011" in manager.channels
    assert "whatsapp:6a06c9c58e6056b24664de25" in manager.channels

    # Each instance bound to its admin_id; bridge_url defaults to the base config
    # (channel's _resolve_ws_url overrides via portfile per connect)
    ch_a = manager.channels["whatsapp:507f1f77bcf86cd799439011"]
    assert ch_a._admin_id == "507f1f77bcf86cd799439011"

    # Transcription settings propagated from placeholder
    assert ch_a.transcription_provider == "openai"
    assert ch_a.transcription_api_key == "sk-test"
    assert ch_a.transcription_language == "zh"


def test_registry_expand_creates_channels_even_without_portfile(tmp_wa_root):
    """expand() no longer waits for bridge — channels are created and their
    own reconnect loop handles bridge availability via _resolve_ws_url."""
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    # NO portfile written — bridge not started yet

    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()

    # Channel still gets created; will retry connecting until bridge appears
    assert "whatsapp" not in manager.channels  # placeholder dropped
    assert "whatsapp:507f1f77bcf86cd799439011" in manager.channels


def test_registry_expand_keeps_placeholder_when_no_admin_dirs(tmp_wa_root):
    """No admin dirs → keep placeholder; poll loop will add channels when
    operator mkdir's a new ~/.nanobot/wa/<adminId>/auth/ dir."""
    # No admin dirs, no portfile
    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()

    assert "whatsapp" in manager.channels  # placeholder retained
    assert not any(k.startswith("whatsapp:") for k in manager.channels)


# ---------------------------------------------------------------------------
# WhatsAppMultiTenantRegistry.tick (add/remove)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_tick_adds_new_admin(tmp_wa_root, monkeypatch):
    _write_portfile(tmp_wa_root, 54321)
    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()  # 0 admins yet

    assert not any(k.startswith("whatsapp:") for k in manager.channels)

    # Stub create_task — avoid actually starting the channel (would need real bridge)
    started: list = []

    def fake_create_task(coro):
        started.append(coro)
        coro.close()
        return MagicMock()

    monkeypatch.setattr(wr.asyncio, "create_task", fake_create_task)

    # Operator mkdir's a new admin dir between ticks
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    await registry.tick()

    assert "whatsapp:507f1f77bcf86cd799439011" in manager.channels
    assert len(started) == 1  # channel.start() was scheduled


@pytest.mark.asyncio
async def test_registry_tick_removes_disappeared_admin(tmp_wa_root):
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    _write_portfile(tmp_wa_root, 54321)
    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()

    ch = manager.channels["whatsapp:507f1f77bcf86cd799439011"]
    ch.stop = AsyncMock()

    # Operator rm -rf the admin dir between ticks
    shutil.rmtree(tmp_wa_root / "507f1f77bcf86cd799439011")
    await registry.tick()

    assert "whatsapp:507f1f77bcf86cd799439011" not in manager.channels
    ch.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_tick_inert_when_no_base_config(tmp_wa_root):
    """Registry built from a manager without WhatsApp channel does nothing on tick."""
    manager = MagicMock()
    manager.channels = {}
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    assert registry.base_config is None

    # Even if dirs exist, tick is a no-op
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    _write_portfile(tmp_wa_root, 54321)
    await registry.tick()

    assert manager.channels == {}


@pytest.mark.asyncio
async def test_registry_tick_no_change_when_portfile_disappears(tmp_wa_root):
    """If portfile vanishes mid-run, tick silently no-ops (don't tear down channels)."""
    _make_admin_dir(tmp_wa_root, "507f1f77bcf86cd799439011")
    _write_portfile(tmp_wa_root, 54321)
    manager = _make_manager_with_placeholder()
    registry = wr.WhatsAppMultiTenantRegistry(manager)
    registry.expand()

    snapshot = dict(manager.channels)

    # Portfile disappears (e.g., bridge crashed and restarting)
    (tmp_wa_root / "bridge.port").unlink()
    await registry.tick()

    assert manager.channels == snapshot  # unchanged

"""Integration: per-admin workspace via X-Ola-Acting-As header (Ola N2.8).

Spans api/server.py header extraction → ContextVar → SessionManager
+ MemoryStore + filesystem tools resolving paths to admins/<id>/. Two
concurrent admin requests must land their writes in separate subtrees.

End-to-end pattern matches tests/test_api_acting_as.py — TestClient via
aiohttp test utils, agent.process_direct mocked to capture the per-admin
manager paths the real request would have used.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.agent.admin_context import (
    get_acting_admin_id,
    get_admin_dir_name,
    set_acting_admin_id,
)
from nanobot.agent.memory import MemoryStore
from nanobot.api.server import create_app
from nanobot.session.manager import SessionManager

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)
pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp test utils not installed")


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


@pytest.fixture(autouse=True)
def _reset_admin_context():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


def _build_agent(captures: dict, workspace: Path) -> MagicMock:
    """Agent mock whose process_direct snapshots the per-request paths a
    real SessionManager + MemoryStore would resolve to."""
    agent = MagicMock()
    sessions = SessionManager(workspace)
    memory = MemoryStore(workspace)

    async def fake_process_direct(*_args, **_kwargs):
        admin_id = get_acting_admin_id()
        captures[admin_id] = {
            "admin_dir": get_admin_dir_name(),
            "sessions_dir": str(sessions.sessions_dir),
            "memory_file": str(memory.memory_file),
            "history_file": str(memory.history_file),
            "user_file": str(memory.user_file),
            "soul_file": str(memory.soul_file),
        }
        return "ok"

    agent.process_direct = AsyncMock(side_effect=fake_process_direct)
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_two_admins_resolve_to_disjoint_subtrees(aiohttp_client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captures: dict = {}
    app = create_app(_build_agent(captures, workspace), model_name="test", request_timeout=10.0)
    client = await aiohttp_client(app)

    for admin in ("507f1f77bcf86cd799439011", "507f1f77bcf86cd799439022"):
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Ola-Acting-As": admin},
        )
        assert resp.status == 200

    a = captures["507f1f77bcf86cd799439011"]
    b = captures["507f1f77bcf86cd799439022"]
    assert a["admin_dir"] == "507f1f77bcf86cd799439011"
    assert b["admin_dir"] == "507f1f77bcf86cd799439022"
    assert a["sessions_dir"] != b["sessions_dir"]
    assert a["memory_file"] != b["memory_file"]
    assert a["history_file"] != b["history_file"]
    assert a["user_file"] != b["user_file"]
    # SOUL.md stays global.
    assert a["soul_file"] == b["soul_file"] == str(workspace / "SOUL.md")
    # Sessions land under workspace/admins/<id>/sessions/, not workspace/sessions/.
    assert str(workspace / "sessions") not in a["sessions_dir"] or "admins" in a["sessions_dir"]
    assert "admins/507f1f77bcf86cd799439011" in a["sessions_dir"]
    assert "admins/507f1f77bcf86cd799439022" in b["sessions_dir"]


@pytest.mark.asyncio
async def test_concurrent_requests_no_cross_contamination(aiohttp_client, tmp_path):
    """Two requests fired in parallel each see their own ContextVar value
    in the agent mock (no asyncio task interleaving leak)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captures: dict = {}
    app = create_app(_build_agent(captures, workspace), model_name="test", request_timeout=10.0)
    client = await aiohttp_client(app)

    async def _send(admin_id: str) -> None:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": f"hi from {admin_id}"}]},
            headers={"X-Ola-Acting-As": admin_id},
        )
        assert resp.status == 200

    admins = [f"admin-{i:024d}" for i in range(8)]
    await asyncio.gather(*[_send(a) for a in admins])

    for admin in admins:
        cap = captures[admin]
        assert cap["admin_dir"] == admin
        assert f"admins/{admin}" in cap["sessions_dir"]
        assert f"admins/{admin}" in cap["memory_file"]


@pytest.mark.asyncio
async def test_no_header_falls_back_to_system_dir(aiohttp_client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captures: dict = {}
    app = create_app(_build_agent(captures, workspace), model_name="test", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    cap = captures[None]
    assert cap["admin_dir"] == "_system"
    assert "admins/_system/sessions" in cap["sessions_dir"]
    assert "admins/_system/memory/MEMORY.md" in cap["memory_file"]

"""api/server.py extracts X-Ola-Acting-As → set_acting_as contextvar.

The contextvar is read later inside MCPToolWrapper.execute to pick the right
per-identity transport from MCPClientPool. The end-to-end header propagation
through the MCP transport is covered by
`tests/agent/test_mcp_pool_real_transport.py`; this file only verifies that
the api/server.py entry point sets the contextvar correctly under sync,
async, and concurrent conditions.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.agent.tools.mcp import get_acting_as, set_acting_as
from nanobot.api.server import create_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


def _agent_capturing_acting_as(captured: dict) -> MagicMock:
    agent = MagicMock()

    async def fake_process_direct(*_args, **_kwargs):
        captured["seen"] = get_acting_as()
        return "ok"

    agent.process_direct = AsyncMock(side_effect=fake_process_direct)
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()
    return agent


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
def _reset_acting_as_between_tests():
    set_acting_as(None)
    yield
    set_acting_as(None)


@pytest.mark.asyncio
async def test_header_with_admin_id_propagates_to_contextvar(aiohttp_client):
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": "507f1f77bcf86cd799439011"},
    )
    assert resp.status == 200
    assert captured["seen"] == "507f1f77bcf86cd799439011"


@pytest.mark.asyncio
async def test_no_header_leaves_contextvar_none(aiohttp_client):
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    assert captured["seen"] is None


@pytest.mark.asyncio
async def test_empty_header_normalized_to_none(aiohttp_client):
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": ""},
    )
    assert resp.status == 200
    assert captured["seen"] is None


@pytest.mark.asyncio
async def test_whitespace_header_normalized_to_none(aiohttp_client):
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": "   "},
    )
    assert resp.status == 200
    assert captured["seen"] is None


@pytest.mark.asyncio
async def test_header_with_surrounding_whitespace_is_trimmed(aiohttp_client):
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": "  abcdef0123456789abcdef01  "},
    )
    assert resp.status == 200
    assert captured["seen"] == "abcdef0123456789abcdef01"


@pytest.mark.asyncio
async def test_concurrent_requests_isolated_by_admin_id(aiohttp_client):
    captured_a: dict = {}
    captured_b: dict = {}

    async def make_agent_for(captured: dict) -> MagicMock:
        agent = MagicMock()

        async def fake_process_direct(*_args, **_kwargs):
            captured["seen_before_yield"] = get_acting_as()
            await asyncio.sleep(0.05)
            captured["seen_after_yield"] = get_acting_as()
            return "ok"

        agent.process_direct = AsyncMock(side_effect=fake_process_direct)
        agent._connect_mcp = AsyncMock()
        agent.close_mcp = AsyncMock()
        return agent

    agent_a = await make_agent_for(captured_a)
    agent_b = await make_agent_for(captured_b)
    app_a = create_app(agent_a, model_name="test-model", request_timeout=10.0)
    app_b = create_app(agent_b, model_name="test-model", request_timeout=10.0)
    client_a = await aiohttp_client(app_a)
    client_b = await aiohttp_client(app_b)

    async def post(client, header_val: str):
        return await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Ola-Acting-As": header_val},
        )

    resp_a, resp_b = await asyncio.gather(
        post(client_a, "admin-A"),
        post(client_b, "admin-B"),
    )
    assert resp_a.status == 200 and resp_b.status == 200
    assert captured_a["seen_before_yield"] == "admin-A"
    assert captured_a["seen_after_yield"] == "admin-A"
    assert captured_b["seen_before_yield"] == "admin-B"
    assert captured_b["seen_after_yield"] == "admin-B"


@pytest.mark.asyncio
async def test_sequential_requests_each_see_own_header(aiohttp_client):
    captured_list: list[dict] = []

    agent = MagicMock()

    async def fake_process_direct(*_args, **_kwargs):
        captured: dict = {"seen": get_acting_as()}
        captured_list.append(captured)
        return "ok"

    agent.process_direct = AsyncMock(side_effect=fake_process_direct)
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": "admin-A"},
    )
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-Acting-As": "admin-B"},
    )
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert [c["seen"] for c in captured_list] == ["admin-A", "admin-B", None]

"""ISO5b (Ola CRM #185) — chat completions extracts X-Ola-Acting-As header.

Verifies:
  - With header → contextvar holds the trimmed value during request handling
  - Without header → contextvar stays None (back-compat)
  - Whitespace / empty header → normalized to None (no leak via header injection)
  - Concurrent requests with different headers don't leak across tasks
  - The hook (mcp._inject_acting_as_hook) sees the right value when invoked
    inside agent.process_direct (the actual call path used by the agent loop)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from nanobot.agent.tools.mcp import (
    _inject_acting_as_hook,
    get_acting_as,
    set_acting_as,
)
from nanobot.api.server import create_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


# ---------------------------------------------------------------------------
# Mock agent that captures the contextvar value DURING process_direct.
# This simulates "the agent loop runs, calls MCP tools, which trigger the
# hook" — without spinning a real MCP server.
# ---------------------------------------------------------------------------


def _agent_capturing_acting_as(captured: dict) -> MagicMock:
    agent = MagicMock()

    async def fake_process_direct(*_args, **_kwargs):
        # This runs in the same task as handle_chat_completions, so the
        # contextvar set in handle_chat_completions is visible here.
        captured["seen"] = get_acting_as()
        # Also exercise the actual hook path (what MCP HTTP calls do).
        req = httpx.Request("POST", "http://127.0.0.1:8889/mcp")
        await _inject_acting_as_hook(req)
        captured["header"] = req.headers.get("X-Acting-As")
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
    """Hard-reset the contextvar between tests so leakage from a prior test
    can't masquerade as success in the next."""
    set_acting_as(None)
    yield
    set_acting_as(None)


# ---------------------------------------------------------------------------
# Header → contextvar tests
# ---------------------------------------------------------------------------


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
    assert captured["header"] == "507f1f77bcf86cd799439011"


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
    assert captured["header"] is None  # hook MUST NOT inject header


@pytest.mark.asyncio
async def test_empty_header_normalized_to_none(aiohttp_client):
    """Empty X-Ola-Acting-As must not propagate as ''."""
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
    assert captured["header"] is None


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


# ---------------------------------------------------------------------------
# Concurrent isolation — the safety property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests_isolated_by_admin_id(aiohttp_client):
    """Two simultaneous chat completions with different X-Ola-Acting-As
    must not leak identity to each other."""

    captured_a: dict = {}
    captured_b: dict = {}

    # Each request needs its own session_lock; agent's process_direct serializes
    # within a single session. Use distinct session_id values so the two calls
    # don't queue behind one lock and serialize.
    async def make_agent_for(captured: dict) -> MagicMock:
        agent = MagicMock()

        async def fake_process_direct(*_args, **_kwargs):
            captured["seen_before_yield"] = get_acting_as()
            await asyncio.sleep(0.05)  # let the other request interleave
            captured["seen_after_yield"] = get_acting_as()
            req = httpx.Request("POST", "http://127.0.0.1:8889/mcp")
            await _inject_acting_as_hook(req)
            captured["header"] = req.headers.get("X-Acting-As")
            return "ok"

        agent.process_direct = AsyncMock(side_effect=fake_process_direct)
        agent._connect_mcp = AsyncMock()
        agent.close_mcp = AsyncMock()
        return agent

    # Two independent apps to ensure two independent session locks.
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
    assert captured_a["header"] == "admin-A"
    assert captured_b["seen_before_yield"] == "admin-B"
    assert captured_b["seen_after_yield"] == "admin-B"
    assert captured_b["header"] == "admin-B"


# ---------------------------------------------------------------------------
# Sequential same-client requests — no stale value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_requests_each_see_own_header(aiohttp_client):
    """Request 1 sets admin-A; request 2 with admin-B must NOT see leftover A."""
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
    # request 3 with NO header — must see None (not leftover B)
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert [c["seen"] for c in captured_list] == ["admin-A", "admin-B", None]


# ---------------------------------------------------------------------------
# Header case sensitivity (HTTP headers are case-insensitive per RFC)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_header_case_insensitive(aiohttp_client):
    """aiohttp normalizes header lookup; both casings must work."""
    captured: dict = {}
    agent = _agent_capturing_acting_as(captured)
    app = create_app(agent, model_name="test-model", request_timeout=10.0)
    client = await aiohttp_client(app)

    # lowercase
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-ola-acting-as": "admin-lower"},
    )
    assert captured["seen"] == "admin-lower"

    # mixed case
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Ola-ACTING-as": "admin-mixed"},
    )
    assert captured["seen"] == "admin-mixed"

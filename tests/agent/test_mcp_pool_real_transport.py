"""Real-transport tests for X-Acting-As header propagation through the
MCP streamableHttp client.

This file exercises the actual SDK transport (`streamable_http_client`)
against a live aiohttp fake MCP server. It exists to catch the bug class
that pure-mock tests cannot: contextvar inheritance across SDK-spawned
transport tasks (`post_writer` / `handle_request_async`).

Pre-fix expectation (Phase ISO red):
  test_concurrent_calls_isolate FAILS — all concurrent tool calls collapse
  to a single X-Acting-As value (the one in scope at connect time), proving
  the _post_writer task contextvar freeze.

Post-fix expectation:
  All four tests PASS — every outgoing MCP HTTP request carries the
  X-Acting-As of the calling task, not the connect-time task.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any

import pytest
from aiohttp import web

from nanobot.agent.tools.mcp import (
    MCPToolWrapper,
    _acting_as_ctx,
    connect_mcp_servers,
    set_acting_as,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig


# ---------------------------------------------------------------------------
# Fake MCP server (aiohttp) — speaks the minimum streamableHttp dialect
# the SDK client expects for initialize / tools/list / tools/call.
# ---------------------------------------------------------------------------


class FakeMCPServer:
    def __init__(self) -> None:
        self.received: list[dict[str, str]] = []  # one entry per request
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self.url: str = ""

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.received.append(
            {
                "method": body.get("method", ""),
                "id": body.get("id"),
                "X-Acting-As": request.headers.get("X-Acting-As"),
                "Authorization": request.headers.get("Authorization"),
            }
        )

        method = body.get("method")
        msg_id = body.get("id")

        if method == "notifications/initialized":
            return web.Response(status=202)

        if method == "initialize":
            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1.0"},
                },
            }
        elif method == "tools/list":
            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "dummy",
                            "description": "echo for tests",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        }
                    ]
                },
            }
        elif method == "resources/list":
            payload = {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}
        elif method == "prompts/list":
            payload = {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}
        elif method == "tools/call":
            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }
        else:
            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }

        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps(payload),
        )

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/mcp", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        sock = self._site._server.sockets[0]  # type: ignore[union-attr]
        port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self.url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


@pytest.fixture
async def fake_mcp() -> Any:
    server = FakeMCPServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
async def connected_pool(fake_mcp: FakeMCPServer):
    set_acting_as(None)
    registry = ToolRegistry()
    cfg = MCPServerConfig(
        type="streamableHttp",
        url=fake_mcp.url,
        headers={"Authorization": "Bearer test-token"},
        tool_timeout=10,
        enabled_tools=["*"],
    )
    pool, _succeeded = await connect_mcp_servers({"fake": cfg}, registry)

    # streamable_http_client opens an anyio task group on first use; if that
    # transport was opened in a now-dead asyncio.gather subtask, anyio raises
    # "Attempted to exit cancel scope in a different task" during teardown.
    # That noise is real for the test fixture but harmless — production
    # close_mcp runs at process exit. Suppress it so it doesn't flip the test.
    loop = asyncio.get_event_loop()
    prev_handler = loop.get_exception_handler()

    def _filter(loop, ctx):
        msg = ctx.get("message", "")
        if (
            "closing of asynchronous generator" in msg
            or "unhandled errors in a TaskGroup" in msg
            or "different task than it was entered in" in msg
        ):
            return
        (prev_handler or loop.default_exception_handler)(ctx)

    loop.set_exception_handler(_filter)
    try:
        yield registry, pool
    finally:
        try:
            await pool.close()
        except (RuntimeError, BaseExceptionGroup, Exception):
            pass
        loop.set_exception_handler(prev_handler)


def _tool_call_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["method"] == "tools/call"]


# ---------------------------------------------------------------------------
# 1. Single-task propagation — the simplest happy path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_acting_as_propagates(fake_mcp: FakeMCPServer, connected_pool):
    registry, _ = connected_pool
    tool = registry.get("mcp_fake_dummy")
    assert isinstance(tool, MCPToolWrapper)

    set_acting_as("admin_alpha")
    out = await tool.execute()
    assert out == "ok"

    calls = _tool_call_records(fake_mcp.received)
    assert len(calls) == 1, calls
    assert calls[0]["X-Acting-As"] == "admin_alpha", (
        f"expected 'admin_alpha', got {calls[0]['X-Acting-As']!r} — "
        "transport task is not seeing caller's contextvar"
    )


# ---------------------------------------------------------------------------
# 2. Two serial calls in the same task — second must not be locked to first.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_serial_calls_isolate(fake_mcp: FakeMCPServer, connected_pool):
    registry, _ = connected_pool
    tool = registry.get("mcp_fake_dummy")

    set_acting_as("admin_first")
    await tool.execute()
    set_acting_as("admin_second")
    await tool.execute()

    calls = _tool_call_records(fake_mcp.received)
    assert [c["X-Acting-As"] for c in calls] == ["admin_first", "admin_second"], (
        f"second call collapsed to first identity: {calls}"
    )


# ---------------------------------------------------------------------------
# 3. Concurrent tasks — the canonical failing case for bug 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_isolate(fake_mcp: FakeMCPServer, connected_pool):
    registry, _ = connected_pool
    tool = registry.get("mcp_fake_dummy")
    n = 10

    async def call_with(admin_id: str) -> None:
        set_acting_as(admin_id)
        await tool.execute()

    await asyncio.gather(*[call_with(f"admin_{i}") for i in range(n)])

    calls = _tool_call_records(fake_mcp.received)
    assert len(calls) == n, f"expected {n} tool/call records, got {len(calls)}"

    observed = sorted(c["X-Acting-As"] for c in calls if c["X-Acting-As"])
    expected = sorted(f"admin_{i}" for i in range(n))
    assert observed == expected, (
        "concurrent calls collapsed to a single (or wrong) identity. "
        f"observed={observed} expected={expected} — proves _post_writer "
        "task contextvar freeze (Phase ISO bug 1)."
    )


# ---------------------------------------------------------------------------
# 4. Heavy concurrency — N=50 stress, mimics email burst load.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_isolate_n50(fake_mcp: FakeMCPServer, connected_pool):
    registry, pool = connected_pool
    tool = registry.get("mcp_fake_dummy")
    n = 50
    # Reuse the same identity across ~half the tasks so the pool's session
    # cache is exercised under contention; the other half are unique.
    unique_ids = 25
    ids = [f"admin_{i % unique_ids}" for i in range(n)]

    async def call_with(admin_id: str) -> None:
        set_acting_as(admin_id)
        await tool.execute()

    await asyncio.gather(*[call_with(a) for a in ids])

    calls = _tool_call_records(fake_mcp.received)
    assert len(calls) == n
    observed = sorted(c["X-Acting-As"] for c in calls if c["X-Acting-As"])
    expected = sorted(ids)
    assert observed == expected, (
        f"N=50 stress: observed {len(observed)} headers but the multiset "
        f"does not match the per-task acting_as. observed={observed[:5]}... "
        f"expected={expected[:5]}..."
    )

    # Pool reuse: exactly `unique_ids` distinct acting_as keys (plus the
    # discovery session keyed by None) — proves second admin_X call hits the
    # cached session rather than reopening a transport every time.
    acting_as_keys = {ident for (_, ident) in pool._sessions.keys() if ident is not None}
    assert acting_as_keys == set(ids), (
        f"pool created {len(acting_as_keys)} distinct sessions; expected {unique_ids}"
    )


# ---------------------------------------------------------------------------
# 5. None acting-as — header MUST be absent (so backend MCP fails closed).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_acting_as_omits_header(fake_mcp: FakeMCPServer, connected_pool):
    registry, _ = connected_pool
    tool = registry.get("mcp_fake_dummy")

    set_acting_as(None)
    await tool.execute()

    calls = _tool_call_records(fake_mcp.received)
    assert len(calls) == 1
    assert calls[0]["X-Acting-As"] is None, (
        f"X-Acting-As must be absent when contextvar is None, got "
        f"{calls[0]['X-Acting-As']!r}"
    )

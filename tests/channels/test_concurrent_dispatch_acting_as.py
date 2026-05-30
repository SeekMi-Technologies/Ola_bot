"""DEPRECATED — false-positive coverage of the X-Acting-As propagation path.

Kept as a counter-example. These tests sample `_acting_as_ctx` from the
caller's task with a mocked transport, so they cannot observe the SDK
spawning `post_writer` / `handle_request_async` tasks that captured the
contextvar at connect time. Result: the suite was green throughout the
1:20 PT 2026-05-06 incident even though every concurrent dispatch was
silently routing through systemAdmin.

The replacement is `tests/agent/test_mcp_pool_real_transport.py`, which
exercises the real `streamable_http_client` against an aiohttp fake MCP
server and verifies the actual outbound `X-Acting-As` header per call.

Skipped at import time so CI does not surface deceptive green checks.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "DEPRECATED: pure-mock coverage of the SDK transport path is a "
        "false-positive (Phase ISO 2026-05-06). See "
        "tests/agent/test_mcp_pool_real_transport.py for the real-transport "
        "replacement."
    )
)

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import _acting_as_ctx
from nanobot.bus.events import InboundMessage


def _bare_loop() -> AgentLoop:
    """Construct a minimal AgentLoop with only the fields _dispatch touches."""
    loop = AgentLoop.__new__(AgentLoop)
    loop._mcp_connected = True
    loop._mcp_connecting = False
    loop._mcp_servers = {}
    loop._mcp_stacks = []
    loop._session_locks = {}
    loop._concurrency_gate = None
    loop._pending_queues = {}
    loop._active_tasks = {}
    loop.sessions = MagicMock()
    loop.bus = AsyncMock()
    loop.commands = MagicMock(
        is_priority=lambda x: False,
        is_dispatchable_command=lambda x: False,
    )
    loop.auto_compact = MagicMock()
    loop.tools = MagicMock()
    loop._effective_session_key = lambda m: m.chat_id
    return loop


def _make_msg(chat_id: str, admin_id: str) -> InboundMessage:
    return InboundMessage(
        channel="email",
        sender_id=chat_id,
        chat_id=chat_id,
        content="...",
        media=[],
        metadata={"_acting_as": admin_id},
    )


@pytest.mark.asyncio
async def test_two_concurrent_dispatch_isolate_acting_as():
    """Two interleaved _dispatch tasks must not see each other's acting-as.

    Mirrors the 1:20 PT 5-6 scenario: Will's email and zyd's email landed in
    the same polling cycle and were processed by two concurrent _dispatch
    tasks. Each task's MCP requests must carry only that task's admin_id.
    """
    captured: dict[str, list[str | None]] = {}

    async def fake_process_message(self, msg, *args, **kwargs):
        cid = msg.chat_id
        captured.setdefault(cid, []).append(_acting_as_ctx.get())
        # Yield to scheduler several times to simulate LLM iteration latency
        for _ in range(5):
            await asyncio.sleep(0.001)
            captured[cid].append(_acting_as_ctx.get())
        return None

    loop = _bare_loop()

    msgs = [
        _make_msg("will@example.com", "WILL_ADMIN_ID"),
        _make_msg("zyd@example.com", "ZYD_ADMIN_ID"),
    ]

    with patch.object(AgentLoop, "_process_message", new=fake_process_message):
        # Production path: agent.loop.run() spawns _dispatch via create_task.
        # create_task forks the parent context, so each task has its own
        # independent ContextVar values.
        tasks = [asyncio.create_task(loop._dispatch(m)) for m in msgs]
        await asyncio.gather(*tasks)

    will_samples = captured.get("will@example.com", [])
    zyd_samples = captured.get("zyd@example.com", [])

    assert will_samples, "will task never reached _process_message"
    assert zyd_samples, "zyd task never reached _process_message"
    assert all(v == "WILL_ADMIN_ID" for v in will_samples), (
        f"will task saw foreign acting_as: {will_samples}"
    )
    assert all(v == "ZYD_ADMIN_ID" for v in zyd_samples), (
        f"zyd task saw foreign acting_as: {zyd_samples}"
    )


@pytest.mark.asyncio
async def test_n_concurrent_dispatch_stress():
    """Spawn 20 concurrent _dispatch tasks with different acting-as values.
    Sample contextvar across many awaits per task — none must bleed."""
    N = 20
    captured: dict[str, list[str | None]] = {}

    async def fake_process_message(self, msg, *args, **kwargs):
        cid = msg.chat_id
        captured.setdefault(cid, []).append(_acting_as_ctx.get())
        for _ in range(8):
            await asyncio.sleep(0.0005)
            captured[cid].append(_acting_as_ctx.get())
        return None

    loop = _bare_loop()
    msgs = [_make_msg(f"u{i}@x.com", f"ADMIN_{i:03d}") for i in range(N)]

    with patch.object(AgentLoop, "_process_message", new=fake_process_message):
        tasks = [asyncio.create_task(loop._dispatch(m)) for m in msgs]
        await asyncio.gather(*tasks)

    for i in range(N):
        cid = f"u{i}@x.com"
        expected = f"ADMIN_{i:03d}"
        samples = captured.get(cid, [])
        assert samples, f"task {cid} never reached _process_message"
        assert all(v == expected for v in samples), (
            f"task {cid} saw foreign acting_as: {samples}"
        )


@pytest.mark.asyncio
async def test_dispatch_acting_as_survives_connect_mcp_call():
    """_dispatch awaits _connect_mcp before _process_message. Even if two
    tasks race through _connect_mcp (one wins, one early-returns), each
    task's contextvar must persist correctly afterward."""
    captured: dict[str, str | None] = {}
    connect_calls: list[str] = []

    # Slow _connect_mcp that simulates real connection time
    async def slow_connect_mcp(self):
        if self._mcp_connected or self._mcp_connecting:
            return
        self._mcp_connecting = True
        try:
            connect_calls.append(_acting_as_ctx.get() or "<none>")
            await asyncio.sleep(0.01)
            self._mcp_connected = True
        finally:
            self._mcp_connecting = False

    async def fake_process_message(self, msg, *args, **kwargs):
        # Read contextvar AFTER _connect_mcp returned
        captured[msg.chat_id] = _acting_as_ctx.get()
        return None

    loop = _bare_loop()
    loop._mcp_connected = False  # force connect on first dispatch
    loop._mcp_servers = {"ola_crm": {}}  # non-empty triggers connect

    msgs = [
        _make_msg("a@x.com", "ADMIN_A"),
        _make_msg("b@x.com", "ADMIN_B"),
        _make_msg("c@x.com", "ADMIN_C"),
    ]

    with (
        patch.object(AgentLoop, "_connect_mcp", new=slow_connect_mcp),
        patch.object(AgentLoop, "_process_message", new=fake_process_message),
    ):
        tasks = [asyncio.create_task(loop._dispatch(m)) for m in msgs]
        await asyncio.gather(*tasks)

    assert captured == {
        "a@x.com": "ADMIN_A",
        "b@x.com": "ADMIN_B",
        "c@x.com": "ADMIN_C",
    }, f"acting-as bled across _connect_mcp race: {captured}"


@pytest.mark.asyncio
async def test_dispatch_simulating_httpx_request_hook_concurrent():
    """Most realistic test — simulates the actual MCP HTTP request path.
    Mimics what _inject_acting_as_hook does at httpx send time: read the
    contextvar in the calling task. Multiple concurrent dispatches each
    make multiple "MCP requests" with delays interleaved."""
    headers_seen: dict[str, list[str | None]] = {}

    async def fake_mcp_request(chat_id: str, n: int):
        # Mimic _inject_acting_as_hook: read contextvar at request time
        for _ in range(n):
            await asyncio.sleep(0.0008)
            headers_seen.setdefault(chat_id, []).append(_acting_as_ctx.get())

    async def fake_process_message(self, msg, *args, **kwargs):
        # Simulate agent runner making multiple MCP tool calls
        await fake_mcp_request(msg.chat_id, 6)
        return None

    loop = _bare_loop()
    msgs = [_make_msg(f"u{i}@x.com", f"ADMIN_{i}") for i in range(10)]

    with patch.object(AgentLoop, "_process_message", new=fake_process_message):
        tasks = [asyncio.create_task(loop._dispatch(m)) for m in msgs]
        await asyncio.gather(*tasks)

    for i in range(10):
        cid = f"u{i}@x.com"
        expected = f"ADMIN_{i}"
        seen = headers_seen.get(cid, [])
        assert seen, f"no requests captured for {cid}"
        assert all(h == expected for h in seen), (
            f"task {cid} sent foreign X-Acting-As: {seen}"
        )

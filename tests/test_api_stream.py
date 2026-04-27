"""Tests for SSE streaming support in /v1/chat/completions."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import (
    _sse_chunk,
    _sse_tool_event,
    _SSE_DONE,
    create_app,
)
from nanobot.utils.progress_events import on_progress_accepts_tool_events

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


# ---------------------------------------------------------------------------
# Unit tests for SSE helpers
# ---------------------------------------------------------------------------


def test_sse_chunk_with_delta() -> None:
    raw = _sse_chunk("hello", "test-model", "chatcmpl-abc123")
    line = raw.decode()
    assert line.startswith("data: ")
    payload = json.loads(line[len("data: "):])
    assert payload["id"] == "chatcmpl-abc123"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "test-model"
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_sse_chunk_finish_reason() -> None:
    raw = _sse_chunk("", "m", "id1", finish_reason="stop")
    payload = json.loads(raw.decode().split("data: ", 1)[1])
    assert payload["choices"][0]["delta"] == {}
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_sse_done_format() -> None:
    assert _SSE_DONE == b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Integration tests with aiohttp TestClient
# ---------------------------------------------------------------------------


def _make_streaming_agent(tokens: list[str]) -> MagicMock:
    """Create a mock agent that streams tokens via on_stream callback."""
    agent = MagicMock()
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    async def fake_process_direct(*, content="", media=None, session_key="",
                                  channel="", chat_id="", on_stream=None,
                                  on_stream_end=None, **kwargs):
        if on_stream:
            for token in tokens:
                await on_stream(token)
        if on_stream_end:
            await on_stream_end()
        return " ".join(tokens)

    agent.process_direct = fake_process_direct
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


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_true_returns_sse(aiohttp_client) -> None:
    """stream=true should return text/event-stream with SSE chunks."""
    agent = _make_streaming_agent(["Hello", " world"])
    app = create_app(agent, model_name="test-model")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    assert resp.content_type == "text/event-stream"

    body = await resp.text()
    lines = [l for l in body.split("\n") if l.startswith("data: ")]

    # Should have: 2 token chunks + 1 finish chunk + [DONE]
    data_lines = [l[len("data: "):] for l in lines]
    assert data_lines[-1] == "[DONE]"

    chunks = [json.loads(l) for l in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    # Last chunk before [DONE] should have finish_reason=stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_false_returns_json(aiohttp_client) -> None:
    """stream=false should still return regular JSON response."""
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="normal reply")
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "normal reply"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_default_is_false(aiohttp_client) -> None:
    """Omitting stream should behave like stream=false."""
    agent = MagicMock()
    agent.process_direct = AsyncMock(return_value="default reply")
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["object"] == "chat.completion"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_sse_chunk_ids_are_consistent(aiohttp_client) -> None:
    """All SSE chunks in a single stream should share the same id."""
    agent = _make_streaming_agent(["A", "B", "C"])
    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "go"}], "stream": True},
    )
    body = await resp.text()
    data_lines = [l[len("data: "):] for l in body.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    chunks = [json.loads(l) for l in data_lines]

    chunk_ids = {c["id"] for c in chunks}
    assert len(chunk_ids) == 1, f"Expected single chunk id, got {chunk_ids}"
    assert chunk_ids.pop().startswith("chatcmpl-")


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_passes_on_stream_callbacks(aiohttp_client) -> None:
    """process_direct should be called with on_stream and on_stream_end when streaming."""
    captured_kwargs: dict = {}

    async def fake_process_direct(**kwargs):
        captured_kwargs.update(kwargs)
        if kwargs.get("on_stream_end"):
            await kwargs["on_stream_end"]()
        return "done"

    agent = MagicMock()
    agent.process_direct = fake_process_direct
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    assert captured_kwargs.get("on_stream") is not None
    assert captured_kwargs.get("on_stream_end") is not None


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_with_session_id(aiohttp_client) -> None:
    """Streaming should respect session_id for session key routing."""
    captured_key: str = ""

    async def fake_process_direct(*, session_key="", on_stream=None, on_stream_end=None, **kwargs):
        nonlocal captured_key
        captured_key = session_key
        if on_stream:
            await on_stream("ok")
        if on_stream_end:
            await on_stream_end()
        return "ok"

    agent = MagicMock()
    agent.process_direct = fake_process_direct
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "session_id": "my-session",
        },
    )
    assert resp.status == 200
    assert captured_key == "api:my-session"


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_streaming_backend_failure_does_not_emit_success_terminator(aiohttp_client) -> None:
    """Backend exceptions should not surface as a normal stop+[DONE] stream."""
    agent = MagicMock()

    async def boom(**kwargs):
        raise RuntimeError("backend blew up")

    agent.process_direct = boom
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert resp.status == 200
    body = await resp.text()
    assert '"finish_reason": "stop"' not in body
    assert "[DONE]" not in body


# ---------------------------------------------------------------------------
# Tool-event SSE streaming tests (Ola CRM issue #131, backlog L1)
#
# These cover the 2026-04-27 addition that surfaces tool_events through the
# stream as named `event: tool_event` SSE frames, enabling consumers like
# the CRM olaController to render real-time "thinking" labels.
# ---------------------------------------------------------------------------


def test_sse_tool_event_format() -> None:
    """tool_event frames are named SSE events with JSON payload."""
    event = {"version": 1, "phase": "start", "name": "merch.search"}
    raw = _sse_tool_event(event).decode()
    assert raw.startswith("event: tool_event\n")
    assert raw.endswith("\n\n")
    payload_line = raw.split("\n")[1]
    assert payload_line.startswith("data: ")
    assert json.loads(payload_line[len("data: "):]) == event


def test_on_progress_signature_accepts_tool_events() -> None:
    """The streaming on_progress callback must declare tool_events kwarg so
    nanobot.utils.progress_events.invoke_on_progress() routes structured
    events to it. Mirrors the inspection logic of on_progress_accepts_tool_events.
    """
    async def cb(content: str, *, tool_hint: bool = False,
                 tool_events: list[dict] | None = None) -> None:
        pass

    assert on_progress_accepts_tool_events(cb) is True


def _make_agent_with_tools(
    text_tokens: list[str],
    tool_events_before_text: list[dict],
    *,
    resuming_pause: bool = True,
) -> MagicMock:
    """Mock agent that simulates the real loop's emission order for a
    single-tool turn: on_progress(tool_events=[start]) → on_stream_end(resuming=True)
    → on_progress(tool_events=[end]) → on_stream(token...) → on_stream_end(resuming=False).

    The resuming=True between tools is the bug the L1 fix targets:
    if not handled, the stream terminates before tool_events flush.
    """
    agent = MagicMock()
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    async def fake_process_direct(*, on_stream=None, on_stream_end=None,
                                  on_progress=None, **kwargs):
        if on_progress and tool_events_before_text:
            await on_progress("", tool_hint=True, tool_events=tool_events_before_text)
        if resuming_pause and on_stream_end:
            await on_stream_end(resuming=True)
        if on_stream:
            for tok in text_tokens:
                await on_stream(tok)
        if on_stream_end:
            await on_stream_end(resuming=False)
        return " ".join(text_tokens)

    agent.process_direct = fake_process_direct
    return agent


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_emits_tool_event_frames_before_text(aiohttp_client) -> None:
    """Happy path: tool_event start+end interleave with text deltas in order."""
    tool_events = [
        {"version": 1, "phase": "start", "call_id": "c1", "name": "merch.search"},
        {"version": 1, "phase": "end", "call_id": "c1", "name": "merch.search",
         "result": '{"ok":true,"data":{"found":false}}'},
    ]
    agent = _make_agent_with_tools(["No matches", " for that"], tool_events)
    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "find x"}], "stream": True},
    )
    assert resp.status == 200
    body = await resp.text()

    # 2 tool_event frames present
    assert body.count("event: tool_event\n") == 2
    # Both phases present
    assert '"phase": "start"' in body
    assert '"phase": "end"' in body
    # Text deltas still flow
    assert '"content": "No matches"' in body
    assert '"content": " for that"' in body
    # Stream terminates cleanly with [DONE]
    assert body.rstrip().endswith("[DONE]")
    # tool_event frames come BEFORE text deltas (the L1 ordering invariant)
    first_tool_event_idx = body.index("event: tool_event")
    first_text_idx = body.index('"content": "No matches"')
    assert first_tool_event_idx < first_text_idx


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_resuming_true_does_not_terminate_stream(aiohttp_client) -> None:
    """Regression for L1 bug: on_stream_end(resuming=True) marks a tool-call
    pause, NOT end-of-stream. If we put 'end' on the queue here, the drain
    loop breaks and tool_events that fire AFTER never reach the client.
    """
    captured_calls: list[tuple] = []

    async def fake_process_direct(*, on_stream=None, on_stream_end=None,
                                  on_progress=None, **kwargs):
        # Pause for tools (this is the path that broke in our first L1 attempt).
        await on_stream_end(resuming=True)
        # AFTER the resume-pause, emit tool_events. If resuming=True killed
        # the stream we'd never see these.
        await on_progress("", tool_hint=True, tool_events=[
            {"version": 1, "phase": "end", "name": "merch.search", "call_id": "x"},
        ])
        # Then text and the real end.
        await on_stream("hi")
        await on_stream_end(resuming=False)
        captured_calls.append(("done",))
        return "hi"

    agent = MagicMock()
    agent.process_direct = fake_process_direct
    agent._connect_mcp = AsyncMock()
    agent.close_mcp = AsyncMock()

    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status == 200
    body = await resp.text()
    # The post-pause tool_event must be present — proves resuming=True did NOT
    # short-circuit the drain loop.
    assert "event: tool_event" in body
    assert '"name": "merch.search"' in body
    # And text after the pause also made it through.
    assert '"content": "hi"' in body
    # And we cleanly terminated.
    assert body.rstrip().endswith("[DONE]")
    assert captured_calls == [("done",)]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_stream_pure_text_no_tool_events(aiohttp_client) -> None:
    """Backwards compat: when no tools are called, stream emits ONLY text
    deltas + [DONE] — no spurious event: tool_event frames.
    """
    agent = _make_streaming_agent(["Hello", " world"])
    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    body = await resp.text()
    assert "event: tool_event" not in body
    assert '"content": "Hello"' in body
    assert body.rstrip().endswith("[DONE]")


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_tool_event_with_error_phase_propagates(aiohttp_client) -> None:
    """Mock-coverage for scenario B (tool error): when the agent emits a
    tool_event with phase='error', the SSE frame must surface it intact so
    the client (CRM) can decide how to render. (Real-stack tool-error
    integration is tracked separately as L1-TD.)
    """
    error_event = {
        "version": 1, "phase": "error", "call_id": "e1",
        "name": "merch.search", "error": "MCP server unreachable",
    }
    agent = _make_agent_with_tools(["sorry"], [error_event], resuming_pause=False)
    app = create_app(agent, model_name="m")
    client = await aiohttp_client(app)

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    body = await resp.text()
    assert "event: tool_event" in body
    assert '"phase": "error"' in body
    assert '"error": "MCP server unreachable"' in body

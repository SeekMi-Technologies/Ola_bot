"""Tests for Ola CRM acting-as injection (issue #185 ISO5a).

Covers the contextvar + httpx event hook used to attach a per-request
X-Acting-As header to outgoing MCP HTTP calls without changing function
signatures up the call chain.

Critical property: concurrent asyncio tasks must see isolated contextvar
values — otherwise concurrent chat sessions would leak acting-as identity.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nanobot.agent.tools.mcp import (
    _inject_acting_as_hook,
    get_acting_as,
    set_acting_as,
)


def _make_request() -> httpx.Request:
    """A minimal httpx.Request whose headers can be mutated by the hook."""
    return httpx.Request("POST", "http://127.0.0.1:8889/mcp")


def test_default_value_is_none():
    """Fresh contextvar — no value set, no caller in scope."""
    # Note: in the same pytest session, an earlier test may have set a value.
    # We explicitly clear via set(None) to assert default behavior.
    set_acting_as(None)
    assert get_acting_as() is None


def test_set_then_get_round_trip():
    set_acting_as("507f1f77bcf86cd799439011")
    assert get_acting_as() == "507f1f77bcf86cd799439011"
    set_acting_as(None)  # reset for hygiene


def test_set_none_clears():
    set_acting_as("some-id")
    assert get_acting_as() == "some-id"
    set_acting_as(None)
    assert get_acting_as() is None


@pytest.mark.asyncio
async def test_hook_injects_header_when_value_set():
    set_acting_as("507f1f77bcf86cd799439011")
    req = _make_request()
    await _inject_acting_as_hook(req)
    assert req.headers["X-Acting-As"] == "507f1f77bcf86cd799439011"
    set_acting_as(None)


@pytest.mark.asyncio
async def test_hook_skips_header_when_value_none():
    set_acting_as(None)
    req = _make_request()
    await _inject_acting_as_hook(req)
    assert "X-Acting-As" not in req.headers


@pytest.mark.asyncio
async def test_hook_skips_header_when_value_empty_string():
    """Empty string is falsy — hook should not attach a useless header."""
    set_acting_as("")
    req = _make_request()
    await _inject_acting_as_hook(req)
    assert "X-Acting-As" not in req.headers
    set_acting_as(None)


@pytest.mark.asyncio
async def test_concurrent_tasks_are_isolated():
    """
    Two concurrent chat sessions must NOT see each other's acting-as.

    This is the core safety property — if it fails, askola users would
    leak identity to each other under concurrent load.
    """

    async def session(admin_id: str, observed: list[str]) -> None:
        set_acting_as(admin_id)
        # yield control to let the other task interleave its set_acting_as
        await asyncio.sleep(0)
        # after the yield, our contextvar should still be ours
        observed.append(get_acting_as())
        await asyncio.sleep(0)
        observed.append(get_acting_as())

    a_observed: list[str] = []
    b_observed: list[str] = []
    await asyncio.gather(
        session("admin-A", a_observed),
        session("admin-B", b_observed),
    )

    assert a_observed == ["admin-A", "admin-A"], f"task A leaked: {a_observed}"
    assert b_observed == ["admin-B", "admin-B"], f"task B leaked: {b_observed}"


@pytest.mark.asyncio
async def test_hook_isolation_across_tasks():
    """End-to-end: two tasks each set + invoke the hook, headers must not cross."""

    async def session(admin_id: str) -> str | None:
        set_acting_as(admin_id)
        await asyncio.sleep(0)
        req = _make_request()
        await _inject_acting_as_hook(req)
        return req.headers.get("X-Acting-As")

    a, b = await asyncio.gather(session("admin-A"), session("admin-B"))
    assert a == "admin-A"
    assert b == "admin-B"


@pytest.mark.asyncio
async def test_child_task_inherits_parent_context():
    """asyncio.create_task copies the current context — child sees parent value."""
    set_acting_as("parent-admin")
    seen: list[str | None] = []

    async def child() -> None:
        seen.append(get_acting_as())

    task = asyncio.create_task(child())
    await task
    assert seen == ["parent-admin"]
    set_acting_as(None)


# ---------------------------------------------------------------------------
# Edge cases — input normalization (set_acting_as)
# ---------------------------------------------------------------------------


def test_whitespace_only_value_normalized_to_none():
    """' ' / '   ' / '\\t\\n' must NOT reach the hook as a header value."""
    for raw in ["   ", "\t", "\n", " \t\n "]:
        set_acting_as(raw)
        assert get_acting_as() is None, f"whitespace {raw!r} leaked through"


def test_value_with_surrounding_whitespace_is_trimmed():
    """A real id with leading/trailing whitespace stores trimmed."""
    set_acting_as("  507f1f77bcf86cd799439011  ")
    assert get_acting_as() == "507f1f77bcf86cd799439011"
    set_acting_as(None)


def test_non_string_value_normalized_to_none():
    """int / list / dict etc must not corrupt the contextvar."""
    for raw in [12345, [], {}, object()]:
        set_acting_as(raw)  # type: ignore[arg-type]
        assert get_acting_as() is None, f"non-string {raw!r} leaked through"


def test_typical_objectid_hex_round_trip():
    """24-char hex ObjectId is the realistic shape from CRM Admin._id."""
    set_acting_as("507f1f77bcf86cd799439011")
    assert get_acting_as() == "507f1f77bcf86cd799439011"
    set_acting_as(None)


def test_unicode_value_passes_through():
    """Unlikely but must not crash — let httpx decide if it's a valid header."""
    set_acting_as("管理员-507f1f")
    assert get_acting_as() == "管理员-507f1f"
    set_acting_as(None)


# ---------------------------------------------------------------------------
# Edge cases — set ordering and idempotency
# ---------------------------------------------------------------------------


def test_repeated_set_takes_last_value():
    """Last write wins."""
    set_acting_as("admin-A")
    set_acting_as("admin-B")
    set_acting_as("admin-C")
    assert get_acting_as() == "admin-C"
    set_acting_as(None)


def test_set_to_same_value_is_idempotent():
    set_acting_as("same-id")
    set_acting_as("same-id")
    assert get_acting_as() == "same-id"
    set_acting_as(None)


@pytest.mark.asyncio
async def test_hook_called_twice_on_same_request_is_idempotent():
    """Hook running twice must not duplicate or corrupt the header."""
    set_acting_as("admin-X")
    req = _make_request()
    await _inject_acting_as_hook(req)
    await _inject_acting_as_hook(req)
    assert req.headers["X-Acting-As"] == "admin-X"
    # httpx headers is a multidict-like; ensure we didn't accidentally append
    all_values = req.headers.get_list("X-Acting-As")
    assert all_values == ["admin-X"], f"header duplicated: {all_values}"
    set_acting_as(None)


# ---------------------------------------------------------------------------
# Edge cases — exception safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_in_task_does_not_leak_acting_as():
    """If task A raises mid-flight, task B's contextvar must remain its own."""

    async def raising_session() -> None:
        set_acting_as("admin-A")
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def normal_session(observed: list[str | None]) -> None:
        set_acting_as("admin-B")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        observed.append(get_acting_as())

    b_observed: list[str | None] = []
    results = await asyncio.gather(
        raising_session(),
        normal_session(b_observed),
        return_exceptions=True,
    )
    # task A raised
    assert isinstance(results[0], RuntimeError)
    # task B unaffected — saw its own admin throughout
    assert b_observed == ["admin-B"]


@pytest.mark.asyncio
async def test_hook_does_not_swallow_unexpected_errors_in_request_object():
    """A broken request mock should not silently no-op — fail loudly."""

    class FrozenHeaders:
        def __setitem__(self, k, v):
            raise TypeError("read-only headers")

    class BrokenRequest:
        headers = FrozenHeaders()

    set_acting_as("admin-X")
    with pytest.raises(TypeError):
        await _inject_acting_as_hook(BrokenRequest())  # type: ignore[arg-type]
    set_acting_as(None)


# ---------------------------------------------------------------------------
# Edge cases — many concurrent tasks (stress)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_concurrent_tasks_no_leakage():
    """50 concurrent sessions, each must only ever see its own admin id."""

    async def session(idx: int) -> bool:
        my_id = f"admin-{idx:03d}"
        set_acting_as(my_id)
        # Yield several times to maximize interleaving with peers.
        for _ in range(5):
            await asyncio.sleep(0)
            if get_acting_as() != my_id:
                return False
        req = _make_request()
        await _inject_acting_as_hook(req)
        return req.headers.get("X-Acting-As") == my_id

    results = await asyncio.gather(*[session(i) for i in range(50)])
    assert all(results), f"isolation broke in {results.count(False)} of 50 tasks"

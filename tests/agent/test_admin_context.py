"""Per-request acting-admin ContextVar — N2 unification (2026-05-19)."""

import asyncio

import pytest

from nanobot.agent.admin_context import (
    SYSTEM_ADMIN_ID,
    get_acting_admin_id,
    get_admin_dir_name,
    set_acting_admin_id,
    with_acting_admin_id,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


def test_default_is_none():
    assert get_acting_admin_id() is None
    assert get_admin_dir_name() == SYSTEM_ADMIN_ID


def test_set_then_get_round_trip():
    set_acting_admin_id("699245d5c692e668ea7ab155")
    assert get_acting_admin_id() == "699245d5c692e668ea7ab155"
    assert get_admin_dir_name() == "699245d5c692e668ea7ab155"


def test_whitespace_normalized_to_none():
    set_acting_admin_id("   ")
    assert get_acting_admin_id() is None


def test_empty_string_normalized_to_none():
    set_acting_admin_id("")
    assert get_acting_admin_id() is None


def test_non_string_normalized_to_none():
    set_acting_admin_id(42)  # type: ignore[arg-type]
    assert get_acting_admin_id() is None
    set_acting_admin_id({"id": "abc"})  # type: ignore[arg-type]
    assert get_acting_admin_id() is None


def test_surrounding_whitespace_trimmed():
    set_acting_admin_id("  abc123  ")
    assert get_acting_admin_id() == "abc123"


def test_with_acting_admin_id_restores_on_exit():
    set_acting_admin_id("outer-admin")
    with with_acting_admin_id("inner-admin"):
        assert get_acting_admin_id() == "inner-admin"
    assert get_acting_admin_id() == "outer-admin"


def test_with_acting_admin_id_restores_on_exception():
    set_acting_admin_id("outer-admin")
    with pytest.raises(RuntimeError):
        with with_acting_admin_id("inner-admin"):
            assert get_acting_admin_id() == "inner-admin"
            raise RuntimeError("boom")
    assert get_acting_admin_id() == "outer-admin"


def test_with_acting_admin_id_nested():
    with with_acting_admin_id("a"):
        with with_acting_admin_id("b"):
            with with_acting_admin_id("c"):
                assert get_acting_admin_id() == "c"
            assert get_acting_admin_id() == "b"
        assert get_acting_admin_id() == "a"
    assert get_acting_admin_id() is None


def test_get_admin_dir_name_falls_back_to_system():
    set_acting_admin_id(None)
    assert get_admin_dir_name() == "_system"


def test_concurrent_tasks_isolated():
    """Two asyncio.create_task() siblings each get their own ContextVar
    copy (per asyncio semantics) — this is the asyncio-safety guarantee
    that lets N2 use ContextVar instead of per-admin instances."""
    captured: dict[str, str | None] = {}

    async def worker(name: str, admin_id: str) -> None:
        set_acting_admin_id(admin_id)
        await asyncio.sleep(0.01)
        captured[name] = get_acting_admin_id()

    async def driver() -> None:
        set_acting_admin_id("driver-original")
        await asyncio.gather(
            worker("A", "admin-A"),
            worker("B", "admin-B"),
            worker("C", "admin-C"),
        )
        captured["driver_after"] = get_acting_admin_id()

    asyncio.run(driver())

    assert captured["A"] == "admin-A"
    assert captured["B"] == "admin-B"
    assert captured["C"] == "admin-C"
    # Driver's own context survives the gather — workers' sets ran in
    # their own task contexts.
    assert captured["driver_after"] == "driver-original"


def test_backward_compat_aliases():
    """set_acting_as / get_acting_as still point at the canonical helpers
    (Phase ISO 2026-05-06 vocabulary preserved)."""
    from nanobot.agent.admin_context import (
        get_acting_as,
        set_acting_as,
        set_acting_admin_id,
        get_acting_admin_id,
    )

    assert set_acting_as is set_acting_admin_id
    assert get_acting_as is get_acting_admin_id


def test_mcp_module_reexports_same_contextvar():
    """nanobot.agent.tools.mcp still exposes _acting_as_ctx / set_acting_as /
    get_acting_as referring to the SAME object — preserves the existing
    MCPClientPool internals + Phase ISO test imports."""
    from nanobot.agent.tools import mcp as mcp_mod
    from nanobot.agent import admin_context as ac_mod

    assert mcp_mod._acting_as_ctx is ac_mod._acting_admin_ctx
    assert mcp_mod.set_acting_as is ac_mod.set_acting_admin_id
    assert mcp_mod.get_acting_as is ac_mod.get_acting_admin_id

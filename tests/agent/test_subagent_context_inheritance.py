"""Subagent ContextVar inheritance via asyncio.create_task (Ola N2.6).

Python's asyncio.create_task calls loop.create_task(coro, context=copy_context())
by default (since 3.7), so a subagent spawned this way inherits the parent's
acting-admin ContextVar. This test guards that invariant — if a future refactor
switches subagent spawn to run_in_executor or a different event loop, the
test fails and the cross-admin filesystem isolation is restored as a hard
blocker rather than discovered in production.
"""

import asyncio

import pytest

from nanobot.agent.admin_context import (
    get_acting_admin_id,
    set_acting_admin_id,
    with_acting_admin_id,
)


@pytest.fixture(autouse=True)
def _reset():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


@pytest.mark.asyncio
async def test_create_task_inherits_acting_admin_contextvar():
    captured = {}

    async def worker(name: str) -> None:
        captured[name] = get_acting_admin_id()

    with with_acting_admin_id("parent-admin"):
        task = asyncio.create_task(worker("subagent"))
        await task

    assert captured["subagent"] == "parent-admin"


@pytest.mark.asyncio
async def test_sibling_subagents_each_see_parent_acting_admin():
    captured: dict[str, str | None] = {}

    async def worker(name: str) -> None:
        await asyncio.sleep(0.01)
        captured[name] = get_acting_admin_id()

    with with_acting_admin_id("admin-X"):
        await asyncio.gather(
            asyncio.create_task(worker("sub1")),
            asyncio.create_task(worker("sub2")),
            asyncio.create_task(worker("sub3")),
        )

    assert captured == {"sub1": "admin-X", "sub2": "admin-X", "sub3": "admin-X"}


@pytest.mark.asyncio
async def test_subagent_mutating_contextvar_does_not_leak_to_parent():
    """Subagent may temporarily switch context (e.g. for nested admin
    operations). The parent's view stays unchanged when the subagent
    completes."""
    async def worker():
        with with_acting_admin_id("nested-admin"):
            return get_acting_admin_id()

    with with_acting_admin_id("parent-admin"):
        nested = await asyncio.create_task(worker())
        assert nested == "nested-admin"
        assert get_acting_admin_id() == "parent-admin"

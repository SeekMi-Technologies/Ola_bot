"""Per-request acting-admin identity (ContextVar).

Set by api/server.py from X-Ola-Acting-As. Read by every admin-scoped
subsystem (sessions, memory, filesystem tools, MCP pool, subagent).

Fallback: None resolves to SYSTEM_ADMIN_ID in get_admin_dir_name() so
anonymous / CLI writes land under workspace/admins/_system/ rather than
falling through to a flat global workspace.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Sentinel admin id used when no admin context is set. All anonymous /
# CLI / channel-without-metadata writes land in workspace/admins/_system/.
SYSTEM_ADMIN_ID = "_system"

_acting_admin_ctx: ContextVar[str | None] = ContextVar(
    "ola_acting_admin", default=None
)


def set_acting_admin_id(value: str | None) -> None:
    """Store the current request's acting-admin id. Empty/whitespace/
    non-string normalize to None so we never propagate junk."""
    if value is None or not isinstance(value, str):
        _acting_admin_ctx.set(None)
        return
    trimmed = value.strip()
    _acting_admin_ctx.set(trimmed if trimmed else None)


def get_acting_admin_id() -> str | None:
    """Current request's acting-admin id, or None if no admin context
    is set (CLI / pre-header / channel anonymous)."""
    return _acting_admin_ctx.get()


def get_admin_dir_name() -> str:
    """Folder-name-safe admin id for workspace path construction. Returns
    the acting-admin id when set, else SYSTEM_ADMIN_ID sentinel."""
    return get_acting_admin_id() or SYSTEM_ADMIN_ID


@contextmanager
def with_acting_admin_id(value: str | None) -> Iterator[None]:
    """Temporarily set the acting-admin id; restore on exit.

    Use case: background tasks that iterate over admins explicitly
    (Consolidator / Dream / manual reprocessing). Asyncio-safe under
    asyncio.create_task because ContextVar.reset uses a token-bound
    restore.
    """
    token = _acting_admin_ctx.set(value)
    try:
        yield
    finally:
        _acting_admin_ctx.reset(token)


# Backward-compat aliases for nanobot.agent.tools.mcp callers.
set_acting_as = set_acting_admin_id
get_acting_as = get_acting_admin_id

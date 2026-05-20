"""#254 cross-admin filesystem leak — repro + N2.4 fix verification.

ContextVar-driven admin scoping prevents admin B's filesystem tools from
reading admin A's sessions / memory / USER.md files. Without N2.4 (the
filesystem-tool callable allowed_dir), `read_file()` on a sibling-admin
path succeeds and leaks contents to admin B.

Repro pattern matches the live demo on duke@olatech.ai 2026-05-19:
admin A writes a session jsonl; admin B's ReadFileTool reads it back.
After N2.4 the read returns "Error: Path ... outside allowed directory"
and the contents stay private.
"""

import asyncio
from pathlib import Path

import pytest

from nanobot.agent.admin_context import (
    set_acting_admin_id,
    with_acting_admin_id,
)
from nanobot.agent.tools.filesystem import (
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nanobot.agent.tools.search import GlobTool, GrepTool


@pytest.fixture(autouse=True)
def _reset():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def admin_scoped_tools(workspace: Path):
    """Tools registered the way AgentLoop._register_default_tools wires
    them for Ola N2: workspace+allowed_dir resolve from ContextVar."""
    from nanobot.agent.admin_context import get_admin_dir_name

    def admin_workspace() -> Path:
        return workspace / "admins" / get_admin_dir_name()

    return {
        "read": ReadFileTool(workspace=admin_workspace, allowed_dir=admin_workspace),
        "write": WriteFileTool(workspace=admin_workspace, allowed_dir=admin_workspace),
        "list": ListDirTool(workspace=admin_workspace, allowed_dir=admin_workspace),
        "glob": GlobTool(workspace=admin_workspace, allowed_dir=admin_workspace),
        "grep": GrepTool(workspace=admin_workspace, allowed_dir=admin_workspace),
    }


def _run(coro):
    return asyncio.run(coro)


def _blocked(result) -> bool:
    """Filesystem tools wrap PermissionError as a string of the form
    `Error: Path ... outside allowed directory ...`."""
    s = str(result)
    return s.startswith("Error:") and "outside allowed directory" in s


def test_admin_b_read_cannot_reach_admin_a_session(workspace, admin_scoped_tools):
    a_sessions = workspace / "admins" / "admin-A" / "sessions"
    a_sessions.mkdir(parents=True)
    secret_path = a_sessions / "api_user_admin-A_conv_xxx.jsonl"
    secret_path.write_text("ADMIN A SECRET MESSAGE", encoding="utf-8")

    with with_acting_admin_id("admin-B"):
        result = _run(admin_scoped_tools["read"].execute(path=str(secret_path)))
    assert _blocked(result), f"expected blocked, got: {result}"
    assert "ADMIN A SECRET MESSAGE" not in str(result)


def test_admin_b_grep_cannot_scan_admin_a_sessions(workspace, admin_scoped_tools):
    a_sessions = workspace / "admins" / "admin-A" / "sessions"
    a_sessions.mkdir(parents=True)
    (a_sessions / "a.jsonl").write_text("ADMIN A LEAK MARKER", encoding="utf-8")

    with with_acting_admin_id("admin-B"):
        result = _run(
            admin_scoped_tools["grep"].execute(
                pattern="LEAK MARKER",
                path=str(a_sessions),
            )
        )
    assert _blocked(result), f"expected blocked, got: {result}"
    assert "LEAK MARKER" not in str(result)


def test_admin_b_list_dir_cannot_inspect_admin_a_tree(workspace, admin_scoped_tools):
    a_dir = workspace / "admins" / "admin-A"
    (a_dir / "sessions").mkdir(parents=True)
    (a_dir / "sessions" / "leak.jsonl").write_text("x")

    with with_acting_admin_id("admin-B"):
        result = _run(admin_scoped_tools["list"].execute(path=str(a_dir)))
    assert _blocked(result), f"expected blocked, got: {result}"


def test_admin_b_glob_cannot_walk_admin_a_subtree(workspace, admin_scoped_tools):
    a_sessions = workspace / "admins" / "admin-A" / "sessions"
    a_sessions.mkdir(parents=True)
    (a_sessions / "a.jsonl").write_text("x")

    with with_acting_admin_id("admin-B"):
        result = _run(
            admin_scoped_tools["glob"].execute(
                pattern="*.jsonl",
                path=str(a_sessions),
            )
        )
    assert _blocked(result), f"expected blocked, got: {result}"


def test_admin_b_write_cannot_create_file_in_admin_a_dir(workspace, admin_scoped_tools):
    a_dir = workspace / "admins" / "admin-A"
    a_dir.mkdir(parents=True)

    with with_acting_admin_id("admin-B"):
        result = _run(
            admin_scoped_tools["write"].execute(
                path=str(a_dir / "evil.txt"),
                content="injected by admin B",
            )
        )
    assert _blocked(result), f"expected blocked, got: {result}"
    assert not (a_dir / "evil.txt").exists()


def test_admin_a_can_read_its_own_files(workspace, admin_scoped_tools):
    a_sessions = workspace / "admins" / "admin-A" / "sessions"
    a_sessions.mkdir(parents=True)
    own = a_sessions / "own.jsonl"
    own.write_text("admin A's own conversation", encoding="utf-8")

    with with_acting_admin_id("admin-A"):
        result = _run(admin_scoped_tools["read"].execute(path=str(own)))
    assert "admin A's own conversation" in str(result)


def test_admin_a_grep_only_finds_in_own_subtree(workspace, admin_scoped_tools):
    """Same content in both admins' dirs; grep scoped to acting admin
    surfaces only the acting admin's hit. Grep's default output is the
    matching file path (relative to admin's workspace), not the content,
    so we assert exactly one file is returned."""
    for admin in ("admin-A", "admin-B"):
        d = workspace / "admins" / admin / "sessions"
        d.mkdir(parents=True)
        (d / "x.jsonl").write_text("unique-marker-99", encoding="utf-8")

    with with_acting_admin_id("admin-A"):
        result = _run(
            admin_scoped_tools["grep"].execute(
                pattern="unique-marker-99",
                path=str(workspace / "admins" / "admin-A"),
            )
        )
    out = str(result)
    # Only admin-A's file path is in the output (display path is
    # relative to admin-A's workspace, so just "sessions/x.jsonl").
    assert "sessions/x.jsonl" in out
    assert "admin-B" not in out


def test_relative_path_resolves_under_acting_admin(workspace, admin_scoped_tools):
    """`read_file("memory/foo.txt")` is a relative path; per N2 it lands
    in workspace/admins/<acting>/memory/foo.txt — not the legacy
    workspace/memory/foo.txt."""
    (workspace / "admins" / "admin-A" / "memory").mkdir(parents=True)
    (workspace / "admins" / "admin-A" / "memory" / "foo.txt").write_text("A's note")
    (workspace / "admins" / "admin-B" / "memory").mkdir(parents=True)
    (workspace / "admins" / "admin-B" / "memory" / "foo.txt").write_text("B's note")

    with with_acting_admin_id("admin-A"):
        a_result = _run(admin_scoped_tools["read"].execute(path="memory/foo.txt"))
    with with_acting_admin_id("admin-B"):
        b_result = _run(admin_scoped_tools["read"].execute(path="memory/foo.txt"))
    assert "A's note" in str(a_result)
    assert "B's note" in str(b_result)


def test_extra_allowed_dirs_still_readable(workspace, tmp_path):
    """BUILTIN_SKILLS_DIR-style extra_allowed_dirs work alongside per-admin
    allowed_dir (global read-only skill libraries stay accessible)."""
    from nanobot.agent.admin_context import get_admin_dir_name

    skills_dir = tmp_path / "global_skills"
    skills_dir.mkdir()
    (skills_dir / "shared.md").write_text("# Shared skill")

    def admin_workspace() -> Path:
        return workspace / "admins" / get_admin_dir_name()

    read = ReadFileTool(
        workspace=admin_workspace,
        allowed_dir=admin_workspace,
        extra_allowed_dirs=[skills_dir],
    )

    with with_acting_admin_id("admin-A"):
        result = _run(read.execute(path=str(skills_dir / "shared.md")))
    assert "Shared skill" in str(result)

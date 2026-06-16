"""provision_admin — lazy per-admin workspace skeleton (#354)."""

from pathlib import Path

import pytest

from nanobot.agent.admin_context import SYSTEM_ADMIN_ID
from nanobot.utils.helpers import provision_admin


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Root templates = the Ola files synced on deploy / start-dev (seed source).
    (ws / "USER.md").write_text("# Ola USER template\n- Name: ???\n", encoding="utf-8")
    (ws / "SOUL.md").write_text("# Ola SOUL template\n", encoding="utf-8")
    (ws / "TOOLS.md").write_text("# Ola TOOLS template\n", encoding="utf-8")
    (ws / "AGENTS.md").write_text("# Global AGENTS (security layer)\n", encoding="utf-8")
    return ws


def _admin_dir(ws: Path, admin_id: str) -> Path:
    return ws / "admins" / admin_id


def test_new_admin_fully_provisioned(workspace):
    assert provision_admin(workspace, "admin-A") is True
    d = _admin_dir(workspace, "admin-A")
    assert (d / "USER.md").exists()
    assert (d / "SOUL.md").exists()
    assert (d / "TOOLS.md").exists()
    assert not (d / "AGENTS.md").exists()  # global security layer, never per-admin
    assert (d / "memory" / "MEMORY.md").exists()
    assert (d / "memory" / "history.jsonl").exists()
    assert (d / "sessions").is_dir()
    assert (d / ".provisioned").exists()


def test_persona_files_seeded_from_root_templates(workspace):
    provision_admin(workspace, "admin-A")
    d = _admin_dir(workspace, "admin-A")
    for filename in ("USER.md", "SOUL.md", "TOOLS.md"):
        assert (d / filename).read_text(encoding="utf-8") == (workspace / filename).read_text(encoding="utf-8")


def test_idempotent_second_call_no_writes(workspace):
    assert provision_admin(workspace, "admin-A") is True
    d = _admin_dir(workspace, "admin-A")
    files = [
        d / "USER.md",
        d / "SOUL.md",
        d / "TOOLS.md",
        d / "memory" / "MEMORY.md",
        d / "memory" / "history.jsonl",
        d / ".provisioned",
    ]
    before = {f: f.stat().st_mtime_ns for f in files}

    assert provision_admin(workspace, "admin-A") is False  # marker fast-path
    after = {f: f.stat().st_mtime_ns for f in files}
    assert before == after


def test_existing_file_never_overwritten(workspace):
    d = _admin_dir(workspace, "admin-A")
    (d / "memory").mkdir(parents=True)
    (d / "SOUL.md").write_text("CUSTOM company persona", encoding="utf-8")

    # No .provisioned marker yet → provision fills the rest but keeps SOUL.md.
    assert provision_admin(workspace, "admin-A") is True
    assert (d / "SOUL.md").read_text(encoding="utf-8") == "CUSTOM company persona"
    assert (d / ".provisioned").exists()
    assert (d / "USER.md").exists()
    assert (d / "memory" / "MEMORY.md").exists()


def test_missing_root_template_is_skipped(workspace):
    (workspace / "SOUL.md").unlink()  # no root SOUL template to seed from
    assert provision_admin(workspace, "admin-A") is True
    d = _admin_dir(workspace, "admin-A")
    assert not (d / "SOUL.md").exists()  # skipped, not created empty
    assert (d / "USER.md").exists()  # others still seeded
    assert (d / ".provisioned").exists()


def test_system_admin_is_noop(workspace):
    assert provision_admin(workspace, SYSTEM_ADMIN_ID) is False
    assert not _admin_dir(workspace, SYSTEM_ADMIN_ID).exists()


def test_none_is_noop(workspace):
    assert provision_admin(workspace, None) is False
    assert not (workspace / "admins").exists()


def test_path_traversal_id_creates_no_dir(workspace):
    assert provision_admin(workspace, "../evil") is False
    assert not (workspace / "evil").exists()
    assert not (workspace.parent / "evil").exists()

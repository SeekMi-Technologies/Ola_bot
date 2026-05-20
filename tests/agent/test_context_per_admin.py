"""ContextBuilder per-admin USER.md + identity memory_dir_path (Ola N2.5)."""

from pathlib import Path

import pytest

from nanobot.agent.admin_context import (
    set_acting_admin_id,
    with_acting_admin_id,
)
from nanobot.agent.context import ContextBuilder


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


def test_user_md_loaded_from_acting_admin_dir(workspace):
    (workspace / "admins" / "admin-A").mkdir(parents=True)
    (workspace / "admins" / "admin-A" / "USER.md").write_text(
        "# User Profile\n- Name: Yuandong\n", encoding="utf-8"
    )
    (workspace / "admins" / "admin-B").mkdir(parents=True)
    (workspace / "admins" / "admin-B" / "USER.md").write_text(
        "# User Profile\n- Name: Cici\n", encoding="utf-8"
    )

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        prompt_a = builder.build_system_prompt()
    with with_acting_admin_id("admin-B"):
        prompt_b = builder.build_system_prompt()
    assert "Yuandong" in prompt_a and "Cici" not in prompt_a
    assert "Cici" in prompt_b and "Yuandong" not in prompt_b


def test_global_workspace_files_still_loaded(workspace):
    (workspace / "AGENTS.md").write_text("# Agent guidelines\nGlobal text.", encoding="utf-8")
    (workspace / "SOUL.md").write_text("# Soul\nGlobal doctrine.", encoding="utf-8")

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        prompt = builder.build_system_prompt()
    assert "Global text" in prompt
    assert "Global doctrine" in prompt


def test_identity_shows_admin_scoped_memory_path(workspace):
    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        prompt = builder.build_system_prompt()
    assert "admins/admin-A/memory/MEMORY.md" in prompt
    assert "admins/admin-A/memory/history.jsonl" in prompt
    # Custom skills line stays at workspace root (skills are global).
    assert "{skill-name}" in prompt
    assert str(workspace / "skills") in prompt


def test_memory_md_per_admin_in_prompt(workspace):
    (workspace / "admins" / "admin-A" / "memory").mkdir(parents=True)
    (workspace / "admins" / "admin-A" / "memory" / "MEMORY.md").write_text(
        "# Long-term Memory\nA's specific facts about Apex Industrial.",
        encoding="utf-8",
    )

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        a_prompt = builder.build_system_prompt()
    with with_acting_admin_id("admin-B"):
        b_prompt = builder.build_system_prompt()

    assert "Apex Industrial" in a_prompt
    assert "Apex Industrial" not in b_prompt

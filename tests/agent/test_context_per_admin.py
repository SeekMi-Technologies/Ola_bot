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


# -- #353: SOUL.md / TOOLS.md per-admin override with global fallback ----------


def test_soul_per_admin_override_with_global_fallback(workspace):
    (workspace / "SOUL.md").write_text("# Soul\nGlobal doctrine.", encoding="utf-8")
    (workspace / "admins" / "admin-A").mkdir(parents=True)
    (workspace / "admins" / "admin-A" / "SOUL.md").write_text(
        "# Soul\nA's custom doctrine.", encoding="utf-8"
    )

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        prompt_a = builder.build_system_prompt()
    with with_acting_admin_id("admin-B"):
        prompt_b = builder.build_system_prompt()

    assert "A's custom doctrine" in prompt_a and "Global doctrine" not in prompt_a
    assert "Global doctrine" in prompt_b and "A's custom doctrine" not in prompt_b


def test_global_tools_edit_reaches_non_overridden_admins(workspace):
    # No per-admin TOOLS.md exists: editing the global file must reach an admin
    # immediately on the next read (read-time resolution, no per-admin copies).
    tools = workspace / "TOOLS.md"
    tools.write_text("# Tools\nv1 global tools.", encoding="utf-8")

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        assert "v1 global tools" in builder.build_system_prompt()

    tools.write_text("# Tools\nv2 global tools.", encoding="utf-8")
    with with_acting_admin_id("admin-A"):
        assert "v2 global tools" in builder.build_system_prompt()


def test_agents_md_is_always_global_never_overridden(workspace):
    (workspace / "AGENTS.md").write_text(
        "# Agents\nGlobal security layer.", encoding="utf-8"
    )
    (workspace / "admins" / "admin-A").mkdir(parents=True)
    (workspace / "admins" / "admin-A" / "AGENTS.md").write_text(
        "# Agents\nInjected per-admin override.", encoding="utf-8"
    )

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        prompt = builder.build_system_prompt()

    assert "Global security layer" in prompt
    assert "Injected per-admin override" not in prompt


def test_bootstrap_identical_across_admins_when_no_overrides(workspace):
    # Zero per-admin override files → the bootstrap section is byte-identical for
    # every admin (no behavior change vs the pre-#353 global-only read path).
    (workspace / "AGENTS.md").write_text("# Agents\nG-A.", encoding="utf-8")
    (workspace / "SOUL.md").write_text("# Soul\nG-S.", encoding="utf-8")
    (workspace / "TOOLS.md").write_text("# Tools\nG-T.", encoding="utf-8")

    builder = ContextBuilder(workspace)
    with with_acting_admin_id("admin-A"):
        boot_a = builder._load_bootstrap_files()
    with with_acting_admin_id("admin-B"):
        boot_b = builder._load_bootstrap_files()

    assert boot_a == boot_b
    assert "G-A" in boot_a and "G-S" in boot_a and "G-T" in boot_a

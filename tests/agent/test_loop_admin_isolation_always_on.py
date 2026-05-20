"""AgentLoop._register_default_tools sets per-admin allowed_dir
unconditionally — even when restrict_to_workspace=False (the default).

#254 fix (Ola N2) regression guard: a configuration that turns off the
workspace sandbox must NOT also turn off cross-admin isolation. Reviewed
on PR #4 by @claude (2026-05-19): in the first cut of N2.4, allowed_dir
was None when neither restrict_to_workspace nor exec_config.sandbox was
set, defeating the leak fix for production deployments that don't opt
into the sandbox.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.admin_context import set_acting_admin_id, with_acting_admin_id
from nanobot.agent.loop import AgentLoop


@pytest.fixture(autouse=True)
def _reset():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


def _make_loop(tmp_path: Path, *, restrict: bool = False) -> AgentLoop:
    bus = MagicMock()
    bus.subscribe = MagicMock()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        restrict_to_workspace=restrict,
    )


@pytest.mark.parametrize("restrict", [True, False])
def test_read_file_allowed_dir_is_per_admin_regardless_of_restrict(tmp_path, restrict):
    loop = _make_loop(tmp_path, restrict=restrict)
    rft = loop.tools.get("read_file")
    assert rft is not None, "read_file tool should be registered"
    # _allowed_dir is a callable (admin_workspace factory) regardless of restrict.
    assert callable(rft._allowed_dir), (
        f"restrict_to_workspace={restrict}: allowed_dir must be a "
        "per-admin factory, never None (security invariant)"
    )
    with with_acting_admin_id("admin-A"):
        path_a = rft._allowed_dir()
    with with_acting_admin_id("admin-B"):
        path_b = rft._allowed_dir()
    assert path_a == tmp_path / "admins" / "admin-A"
    assert path_b == tmp_path / "admins" / "admin-B"


@pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file", "list_dir", "glob", "grep"])
def test_every_filesystem_tool_has_per_admin_factory(tmp_path, tool_name):
    loop = _make_loop(tmp_path, restrict=False)
    tool = loop.tools.get(tool_name)
    assert tool is not None, f"{tool_name} tool missing"
    assert callable(tool._allowed_dir), (
        f"{tool_name}: allowed_dir must be a per-admin factory even when "
        "restrict_to_workspace=False"
    )


def test_extra_allowed_dirs_only_present_when_sandbox_gate_set(tmp_path):
    """BUILTIN_SKILLS_DIR is added to read_file's extra_allowed_dirs only
    when restrict_to_workspace or sandbox is on — that gating is still
    correct after fix #2. Cross-admin isolation (allowed_dir) is the
    invariant; BUILTIN_SKILLS_DIR readability is the opt-in."""
    loop_off = _make_loop(tmp_path, restrict=False)
    loop_on = _make_loop(tmp_path, restrict=True)
    assert loop_off.tools.get("read_file")._extra_allowed_dirs is None
    assert loop_on.tools.get("read_file")._extra_allowed_dirs is not None

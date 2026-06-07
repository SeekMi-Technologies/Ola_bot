"""tools.mcp_only locks the main channel-facing agent to MCP tools only.

Ola customer-facing safety: a hostile inbound message must not reach
filesystem/shell/web/spawn/self tools. MCP tools are registered later in
_connect_mcp(); Dream/subagents build their own registry and are unaffected.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.admin_context import set_acting_admin_id
from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import ToolsConfig


@pytest.fixture(autouse=True)
def _reset():
    set_acting_admin_id(None)
    yield
    set_acting_admin_id(None)


def _make_loop(tmp_path: Path, *, mcp_only: bool) -> AgentLoop:
    bus = MagicMock()
    bus.subscribe = MagicMock()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        tools_config=ToolsConfig(mcp_only=mcp_only),
    )


def test_mcp_only_registers_no_builtin_tools(tmp_path):
    loop = _make_loop(tmp_path, mcp_only=True)
    # No MCP server is connected at construction, so the registry is empty.
    assert loop.tools.get_definitions() == []
    for name in ("read_file", "write_file", "edit_file", "list_dir", "glob",
                 "grep", "exec", "web_search", "spawn", "my", "message"):
        assert loop.tools.get(name) is None, f"{name} must not register under mcp_only"


def test_default_keeps_builtin_tools(tmp_path):
    loop = _make_loop(tmp_path, mcp_only=False)
    assert loop.tools.get("read_file") is not None
    assert loop.tools.get("grep") is not None

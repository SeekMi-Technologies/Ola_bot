"""Pure-helper coverage that survives the Phase ISO refactor: input-schema
normalization and Windows stdio launcher wrapping. The execute/connect_mcp_servers
tests in `test_mcp_tool.py` were the false-positive mock-transport pattern
and have been replaced by `tests/agent/test_mcp_pool_real_transport.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import nanobot.agent.tools.mcp as mcp_mod
from nanobot.agent.tools.mcp import (
    MCPToolWrapper,
    _normalize_windows_stdio_command,
)


def _wrap(input_schema: dict) -> MCPToolWrapper:
    tool_def = SimpleNamespace(name="demo", description="d", inputSchema=input_schema)
    pool = SimpleNamespace()  # construction does not exercise the pool
    return MCPToolWrapper(pool, "test", tool_def)


def test_wrapper_preserves_non_nullable_unions() -> None:
    w = _wrap({
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    })
    assert w.parameters["properties"]["value"]["anyOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]


def test_wrapper_normalizes_nullable_property_type_union() -> None:
    w = _wrap({
        "type": "object",
        "properties": {"name": {"type": ["string", "null"]}},
    })
    assert w.parameters["properties"]["name"] == {"type": "string", "nullable": True}


def test_wrapper_normalizes_nullable_property_anyof() -> None:
    w = _wrap({
        "type": "object",
        "properties": {
            "name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "optional name",
            },
        },
    })
    assert w.parameters["properties"]["name"] == {
        "type": "string",
        "description": "optional name",
        "nullable": True,
    }


def test_normalize_windows_stdio_command_is_noop_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "posix", raising=False)
    command, args, env = _normalize_windows_stdio_command(
        "npx", ["-y", "x@latest"], {"FOO": "bar"}
    )
    assert command == "npx"
    assert args == ["-y", "x@latest"]
    assert env == {"FOO": "bar"}


def test_normalize_windows_stdio_command_wraps_npx_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        mcp_mod.shutil,
        "which",
        lambda command, path=None: r"C:\Program Files\nodejs\npx.cmd",
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    command, args, env = _normalize_windows_stdio_command(
        "npx", ["-y", "x@latest"], None
    )
    assert command == r"C:\Windows\System32\cmd.exe"
    assert args == ["/d", "/c", "npx", "-y", "x@latest"]
    assert env is None


def test_normalize_windows_stdio_command_wraps_resolved_cmd_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)

    def _fake_which(command: str, path: str | None = None) -> str:
        assert command == "custom-launcher"
        assert path == r"C:\Tools"
        return r"C:\Tools\custom-launcher.cmd"

    monkeypatch.setattr(mcp_mod.shutil, "which", _fake_which)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    command, args, _env = _normalize_windows_stdio_command(
        "custom-launcher", ["serve"], {"PATH": r"C:\Tools"}
    )
    assert command == r"C:\Windows\System32\cmd.exe"
    assert args == ["/d", "/c", "custom-launcher", "serve"]


def test_normalize_windows_stdio_command_keeps_real_executables_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    command, args, env = _normalize_windows_stdio_command(
        "python.exe", ["-m", "http.server"], {"FOO": "bar"}
    )
    assert command == "python.exe"
    assert args == ["-m", "http.server"]
    assert env == {"FOO": "bar"}


def test_normalize_windows_stdio_command_skips_existing_shells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_mod.os, "name", "nt", raising=False)
    command, args, env = _normalize_windows_stdio_command(
        "cmd.exe", ["/c", "echo", "hello"], None
    )
    assert command == "cmd.exe"
    assert args == ["/c", "echo", "hello"]
    assert env is None

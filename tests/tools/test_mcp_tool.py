"""DEPRECATED — mock-transport coverage of MCPToolWrapper / connect_mcp_servers.

The original suite patched `mcp.client.streamable_http.streamable_http_client`
with a fake context manager and asserted on calls to a fake `session.call_tool`.
That pattern cannot observe the SDK's spawned `post_writer` /
`handle_request_async` tasks and therefore stayed green throughout the
1:20 PT 2026-05-06 X-Acting-As leak.

Replacements:
- Pure-helper coverage (input-schema normalization, Windows stdio command
  wrapping) → `tests/tools/test_mcp_helpers.py`.
- Real-transport coverage of the wrapper / pool path →
  `tests/agent/test_mcp_pool_real_transport.py`.

Skipped at import time so CI does not surface deceptive green checks.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "DEPRECATED: mock-transport coverage of MCPToolWrapper / "
        "connect_mcp_servers (Phase ISO 2026-05-06). See "
        "tests/tools/test_mcp_helpers.py and "
        "tests/agent/test_mcp_pool_real_transport.py."
    )
)

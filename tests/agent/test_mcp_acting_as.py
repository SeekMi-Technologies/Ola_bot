"""DEPRECATED — coverage of the removed `_inject_acting_as_hook` event-hook path.

The acting-as header is no longer attached via an httpx event hook reading
a contextvar at request time. Instead `MCPClientPool` opens a separate
transport per (server, acting_as) and bakes `X-Acting-As` into the
client's connection-level headers (`nanobot/agent/tools/mcp.py`,
Phase ISO 2026-05-06).

Replaced by `tests/agent/test_mcp_pool_real_transport.py`, which drives
the real `streamable_http_client` against a fake MCP server and verifies
the actual outbound headers per call.

Skipped at import time so CI does not surface deceptive green checks.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "DEPRECATED: covers the removed _inject_acting_as_hook contextvar→header "
        "injection path. See tests/agent/test_mcp_pool_real_transport.py."
    )
)

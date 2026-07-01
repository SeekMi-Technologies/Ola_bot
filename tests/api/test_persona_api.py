"""Tests for the internal persona control-plane API."""

from pathlib import Path

import pytest
import pytest_asyncio

from nanobot.api.persona_api import create_persona_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

TOKEN = "secret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "admins").mkdir(parents=True)
    (ws / "SOUL.md").write_text("GLOBAL SOUL\n", encoding="utf-8")
    (ws / "TOOLS.md").write_text("GLOBAL TOOLS\n", encoding="utf-8")
    (ws / "AGENTS.md").write_text("GLOBAL AGENTS\n", encoding="utf-8")
    (ws / "USER.md").write_text("GLOBAL USER TEMPLATE\n", encoding="utf-8")
    # admin-A has a custom SOUL override; admin-B has nothing yet.
    a = ws / "admins" / "admin-A"
    a.mkdir()
    (a / "SOUL.md").write_text("CUSTOM A PERSONA\n", encoding="utf-8")
    (a / "USER.md").write_text("A profile\n", encoding="utf-8")
    (ws / "admins" / "admin-B").mkdir()
    (ws / "admins" / "_system").mkdir()
    return ws


@pytest_asyncio.fixture
async def client(workspace):
    c = TestClient(TestServer(create_persona_app(workspace, TOKEN)))
    await c.start_server()
    yield c
    await c.close()


pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest.mark.asyncio
async def test_health_needs_no_auth(client):
    r = await client.get("/health")
    assert r.status == 200


@pytest.mark.asyncio
async def test_missing_token_rejected(client):
    r = await client.get("/internal/persona")
    assert r.status == 401


@pytest.mark.asyncio
async def test_list_admins_skips_system_and_reports_soul_source(client):
    r = await client.get("/internal/persona", headers=AUTH)
    assert r.status == 200
    admins = {a["adminId"]: a for a in (await r.json())["admins"]}
    assert set(admins) == {"admin-A", "admin-B"}  # _system excluded
    assert admins["admin-A"]["soulSource"] == "override"
    assert admins["admin-A"]["userSource"] == "override"  # fixture gave A a USER.md
    assert admins["admin-B"]["soulSource"] == "global"
    assert admins["admin-B"]["userSource"] == "global"


@pytest.mark.asyncio
async def test_get_returns_override_and_global_sources(client):
    r = await client.get("/internal/persona/admin-A", headers=AUTH)
    files = (await r.json())["files"]
    assert files["SOUL.md"] == {"content": "CUSTOM A PERSONA\n", "source": "override", "editable": True}
    # admin-B has no SOUL override → global fallback
    rb = await client.get("/internal/persona/admin-B", headers=AUTH)
    fb = (await rb.json())["files"]
    assert fb["SOUL.md"]["content"] == "GLOBAL SOUL\n"
    assert fb["SOUL.md"]["source"] == "global"
    # AGENTS/TOOLS are always global + read-only
    assert files["AGENTS.md"]["editable"] is False
    assert files["TOOLS.md"]["editable"] is False


@pytest.mark.asyncio
async def test_put_soul_writes_per_admin_override(client, workspace):
    r = await client.put(
        "/internal/persona/admin-B/SOUL.md", headers=AUTH, json={"content": "B CUSTOM\n"}
    )
    assert r.status == 200
    assert (await r.json())["source"] == "override"
    # written to per-admin dir, not the global root
    assert (workspace / "admins" / "admin-B" / "SOUL.md").read_text() == "B CUSTOM\n"
    assert (workspace / "SOUL.md").read_text() == "GLOBAL SOUL\n"


@pytest.mark.asyncio
async def test_put_agents_forbidden(client):
    r = await client.put(
        "/internal/persona/admin-A/AGENTS.md", headers=AUTH, json={"content": "hack"}
    )
    assert r.status == 403


@pytest.mark.asyncio
async def test_put_tools_forbidden(client):
    r = await client.put(
        "/internal/persona/admin-A/TOOLS.md", headers=AUTH, json={"content": "x"}
    )
    assert r.status == 403


@pytest.mark.asyncio
async def test_system_and_traversal_ids_rejected(client):
    assert (await client.get("/internal/persona/_system", headers=AUTH)).status == 400
    assert (
        await client.put("/internal/persona/..%2Fevil/SOUL.md", headers=AUTH, json={"content": "x"})
    ).status in (400, 404)


@pytest.mark.asyncio
async def test_put_requires_string_content(client):
    r = await client.put("/internal/persona/admin-A/SOUL.md", headers=AUTH, json={"content": 123})
    assert r.status == 400

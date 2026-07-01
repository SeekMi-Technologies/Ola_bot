"""Internal persona control-plane routes.

Lets the Ola devboard read and edit a tenant's per-admin prompt files without
touching the box filesystem directly — nanobot stays the sole owner of its
workspace; callers only ask it to change its own files.

These routes mount onto the existing ``serve`` app (so they need NO new
container / compose service / CD change — they ride the serve port that is
already exposed on Tailscale). ``create_persona_app`` also exposes them as a
standalone app for local dev / a dedicated process if ever wanted.

Surface (all except /health require ``Authorization: Bearer <token>``):
  GET  /internal/persona                     -> list admin ids + SOUL/USER source
  GET  /internal/persona/{adminId}           -> effective SOUL/USER (editable) +
                                                AGENTS/TOOLS (read-only), with source
  PUT  /internal/persona/{adminId}/{file}    -> write per-admin SOUL.md | USER.md

Token resolution (per request, so it is configurable without a restart):
  1. app["persona_token_override"] (standalone --token), else
  2. env PERSONA_API_TOKEN, else
  3. <state-dir>/.persona_token  (workspace.parent/.persona_token) — the file an
     operator drops on the box; different per box, no CD needed.

Boundaries: only SOUL.md/USER.md writable per-admin. AGENTS.md (authority) and
TOOLS.md (shared) are read-only. adminId is path-traversal validated; _system
rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

from aiohttp import web
from loguru import logger

from nanobot.agent.admin_context import SYSTEM_ADMIN_ID, is_valid_admin_id
from nanobot.utils.helpers import _write_text_atomic

# Per-admin, editable via this API.
EDITABLE_FILES = ("SOUL.md", "USER.md")
# Effective content exposed read-only (AGENTS = authority layer; TOOLS = shared).
READONLY_FILES = ("AGENTS.md", "TOOLS.md")


def _workspace(request: web.Request) -> Path:
    return request.app["persona_workspace"]


def _expected_token(app: web.Application) -> str | None:
    override = app.get("persona_token_override")
    if override:
        return override
    env = os.environ.get("PERSONA_API_TOKEN")
    if env:
        return env
    token_file = app["persona_workspace"].parent / ".persona_token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip() or None
    return None


def _authorized(request: web.Request) -> bool:
    expected = _expected_token(request.app)
    if not expected:
        return False
    return request.headers.get("Authorization", "") == f"Bearer {expected}"


def _resolve(workspace: Path, admin_id: str, filename: str) -> tuple[Path, str]:
    """Return (path, source): per-admin override if it exists, else global root."""
    per_admin = workspace / "admins" / admin_id / filename
    if per_admin.exists():
        return per_admin, "override"
    return workspace / filename, "global"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _valid_admin(admin_id: str) -> bool:
    return is_valid_admin_id(admin_id) and admin_id != SYSTEM_ADMIN_ID


def _unauthorized() -> web.Response:
    return web.json_response({"error": "unauthorized"}, status=401)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_list(request: web.Request) -> web.Response:
    if not _authorized(request):
        return _unauthorized()
    admins_dir = _workspace(request) / "admins"
    admins = []
    if admins_dir.is_dir():
        for d in sorted(admins_dir.iterdir()):
            if not d.is_dir() or not _valid_admin(d.name):
                continue
            soul = d / "SOUL.md"
            user = d / "USER.md"
            admins.append(
                {
                    "adminId": d.name,
                    "soulSource": "override" if soul.exists() else "global",
                    "userSource": "override" if user.exists() else "global",
                    "updatedAt": soul.stat().st_mtime if soul.exists() else None,
                }
            )
    return web.json_response({"admins": admins})


async def handle_get(request: web.Request) -> web.Response:
    if not _authorized(request):
        return _unauthorized()
    workspace = _workspace(request)
    admin_id = request.match_info["adminId"]
    if not _valid_admin(admin_id):
        return web.json_response({"error": "invalid adminId"}, status=400)

    files = {}
    for name in EDITABLE_FILES:
        path, source = _resolve(workspace, admin_id, name)
        files[name] = {"content": _read(path), "source": source, "editable": True}
    for name in READONLY_FILES:
        # AGENTS/TOOLS are never per-admin — always the global file.
        files[name] = {"content": _read(workspace / name), "source": "global", "editable": False}
    return web.json_response({"adminId": admin_id, "files": files})


async def handle_put(request: web.Request) -> web.Response:
    if not _authorized(request):
        return _unauthorized()
    workspace = _workspace(request)
    admin_id = request.match_info["adminId"]
    filename = request.match_info["file"]
    if not _valid_admin(admin_id):
        return web.json_response({"error": "invalid adminId"}, status=400)
    if filename not in EDITABLE_FILES:
        return web.json_response(
            {"error": f"{filename} is not editable per-admin (SOUL.md/USER.md only)"},
            status=403,
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    content = body.get("content")
    if not isinstance(content, str):
        return web.json_response({"error": "body.content (string) is required"}, status=400)

    admin_dir = workspace / "admins" / admin_id
    admin_dir.mkdir(parents=True, exist_ok=True)
    dest = admin_dir / filename
    _write_text_atomic(dest, content)
    try:
        os.chown(dest, 1000, 1000)
    except (PermissionError, OSError):
        pass  # dev machines run as the invoking user; container runs as uid 1000
    logger.info("persona PUT admin={} file={} bytes={}", admin_id, filename, len(content))
    return web.json_response(
        {"adminId": admin_id, "file": filename, "bytes": len(content), "source": "override"}
    )


def add_persona_routes(app: web.Application, workspace: Path) -> None:
    """Mount the persona routes onto an existing app (e.g. the serve app).

    Adds no global middleware — each handler checks the bearer token itself, so
    the host app's other routes (chat, health) are untouched.
    """
    app["persona_workspace"] = workspace
    app.router.add_get("/internal/persona", handle_list)
    app.router.add_get("/internal/persona/{adminId}", handle_get)
    app.router.add_put("/internal/persona/{adminId}/{file}", handle_put)


def create_persona_app(workspace: Path, token: str | None = None) -> web.Application:
    """Standalone persona app (local dev / dedicated process)."""
    app = web.Application()
    if token:
        app["persona_token_override"] = token
    app.router.add_get("/health", handle_health)
    add_persona_routes(app, workspace)
    return app

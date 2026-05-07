"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools.

Per-identity transport via MCPClientPool: each acting_as gets its own
httpx.AsyncClient + streamableHttp transport, with X-Acting-As baked into
the client's connection-level headers at creation time. Avoids the SDK's
spawned-task contextvar inheritance pitfall (Phase ISO 2026-05-06).
"""

import asyncio
import os
import shutil
from collections import OrderedDict
from contextlib import AsyncExitStack
from contextvars import ContextVar
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

# Carries the current request's acting-as admin._id from the chat completions
# entry point (api/server.py reads X-Ola-Acting-As) or channel _dispatch
# (msg.metadata['_acting_as']) down to MCPToolWrapper.execute, which keys the
# pool by this value to pick the right transport.
_acting_as_ctx: ContextVar[str | None] = ContextVar("ola_acting_as", default=None)


def set_acting_as(value: str | None) -> None:
    """Store the current request's acting-as identity. Empty/whitespace/non-
    string normalize to None so we never propagate junk."""
    if value is None or not isinstance(value, str):
        _acting_as_ctx.set(None)
        return
    trimmed = value.strip()
    _acting_as_ctx.set(trimmed if trimmed else None)


def get_acting_as() -> str | None:
    return _acting_as_ctx.get()


_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset((
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "ConnectionAbortedError",
    "ConnectionError",
))

_WINDOWS_SHELL_LAUNCHERS: frozenset[str] = frozenset(("npx", "npm", "pnpm", "yarn", "bunx"))


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _windows_command_basename(command: str) -> str:
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()


def _normalize_windows_stdio_command(
    command: str,
    args: list[str] | None,
    env: dict[str, str] | None,
) -> tuple[str, list[str], dict[str, str] | None]:
    normalized_args = list(args or [])
    if os.name != "nt":
        return command, normalized_args, env

    basename = _windows_command_basename(command)
    if basename in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return command, normalized_args, env

    if basename.endswith((".exe", ".com")):
        return command, normalized_args, env

    resolved = shutil.which(command, path=(env or {}).get("PATH")) or command
    resolved_basename = _windows_command_basename(resolved)
    should_wrap = (
        basename in _WINDOWS_SHELL_LAUNCHERS
        or basename.endswith((".cmd", ".bat"))
        or resolved_basename.endswith((".cmd", ".bat"))
    )
    if not should_wrap:
        return command, normalized_args, env

    comspec = (env or {}).get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
    return comspec, ["/d", "/c", command, *normalized_args], env


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop) if isinstance(prop, dict) else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


# ---------------------------------------------------------------------------
# MCPClientPool — per-(server, acting_as) transport sessions
# ---------------------------------------------------------------------------


class MCPClientPool:
    """Per-identity MCP session pool.

    For streamableHttp/sse transports each (server, acting_as) pair gets its
    own httpx.AsyncClient with X-Acting-As baked in at client construction,
    so the SDK's spawned `_post_writer` task inherits the right header at
    spawn time and never needs a contextvar lookup. For stdio transports
    the acting_as key is forced to None — subprocess pipes don't carry HTTP
    headers, so all stdio calls share one transport.

    LRU-capped at `max_size` entries (default 50) — Ola has <20 admins, so
    this is a 2.5x defensive cap against runaway identity churn (e.g. a
    misconfigured agent calling tools with random acting_as values). On
    overflow, the least-recently-used entry's stack is closed and dropped.
    """

    def __init__(self, server_configs: dict, *, max_size: int = 50) -> None:
        self._configs = dict(server_configs)
        # OrderedDict so we can move-to-end on access (LRU) and popitem(last=False)
        # to evict the oldest entry when over `max_size`.
        self._sessions: OrderedDict[tuple[str, str | None], Any] = OrderedDict()
        self._stacks: dict[tuple[str, str | None], AsyncExitStack] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._max_size = max(1, int(max_size))

    def has_server(self, server_name: str) -> bool:
        return server_name in self._configs

    def server_uses_acting_as(self, server_name: str) -> bool:
        cfg = self._configs.get(server_name)
        if cfg is None:
            return False
        ttype = self._infer_transport_type(cfg)
        return ttype in ("streamableHttp", "sse")

    @staticmethod
    def _infer_transport_type(cfg) -> str | None:
        ttype = cfg.type
        if ttype:
            return ttype
        if cfg.command:
            return "stdio"
        if cfg.url:
            return "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
        return None

    async def get_session(self, server_name: str, acting_as: str | None) -> Any:
        if self._closed:
            raise RuntimeError("MCPClientPool is closed")
        if server_name not in self._configs:
            raise KeyError(f"unknown MCP server: {server_name}")

        # stdio transports cannot carry X-Acting-As — collapse to one shared
        # session keyed by None.
        if not self.server_uses_acting_as(server_name):
            acting_as = None

        key = (server_name, acting_as)
        sess = self._sessions.get(key)
        if sess is not None:
            self._sessions.move_to_end(key)  # LRU: mark recently used
            return sess

        async with self._lock:
            sess = self._sessions.get(key)
            if sess is not None:
                self._sessions.move_to_end(key)
                return sess
            sess = await self._open(server_name, acting_as)
            self._sessions[key] = sess
            await self._evict_overflow_locked()
            return sess

    async def _evict_overflow_locked(self) -> None:
        """Drop the least-recently-used entry if pool is over capacity.
        Caller must hold self._lock."""
        while len(self._sessions) > self._max_size:
            evict_key, _ = self._sessions.popitem(last=False)
            stack = self._stacks.pop(evict_key, None)
            if stack is None:
                continue
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                # Cross-task anyio cleanup can raise here; harmless because
                # the entry is gone from both maps regardless.
                logger.debug("MCP pool: evicted entry {} cleanup raised (ignored)", evict_key)
            except Exception as e:
                logger.warning("MCP pool: evicting {}: {}", evict_key, e)

    async def _open(self, server_name: str, acting_as: str | None) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        cfg = self._configs[server_name]
        transport_type = self._infer_transport_type(cfg)
        if transport_type is None:
            raise RuntimeError(
                f"MCP server '{server_name}': no command or url configured"
            )

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if transport_type == "stdio":
                command, args, env = _normalize_windows_stdio_command(
                    cfg.command, cfg.args, cfg.env or None
                )
                params = StdioServerParameters(command=command, args=args, env=env)
                read, write = await stack.enter_async_context(stdio_client(params))
            else:
                base_headers = dict(cfg.headers or {})
                if acting_as:
                    base_headers["X-Acting-As"] = acting_as

                if transport_type == "sse":
                    def httpx_client_factory(
                        headers: dict[str, str] | None = None,
                        timeout: httpx.Timeout | None = None,
                        auth: httpx.Auth | None = None,
                    ) -> httpx.AsyncClient:
                        merged = {
                            "Accept": "application/json, text/event-stream",
                            **base_headers,
                            **(headers or {}),
                        }
                        return httpx.AsyncClient(
                            headers=merged or None,
                            follow_redirects=True,
                            timeout=timeout,
                            auth=auth,
                        )

                    read, write = await stack.enter_async_context(
                        sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                    )
                else:  # streamableHttp
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=base_headers or None,
                            follow_redirects=True,
                            timeout=None,
                        )
                    )
                    read, write, _ = await stack.enter_async_context(
                        streamable_http_client(cfg.url, http_client=http_client)
                    )

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            try:
                await stack.aclose()
            except Exception:
                pass
            raise

        self._stacks[(server_name, acting_as)] = stack
        return session

    async def close(self) -> None:
        self._closed = True
        for key, stack in list(self._stacks.items()):
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP pool: stack {} cleanup error (ignored)", key)
            except Exception as e:
                logger.warning("MCP pool: unexpected error closing {}: {}", key, e)
        self._stacks.clear()
        self._sessions.clear()


# ---------------------------------------------------------------------------
# Tool / Resource / Prompt wrappers — delegate session lookup to pool
# ---------------------------------------------------------------------------


class MCPToolWrapper(Tool):
    def __init__(
        self,
        pool: MCPClientPool,
        server_name: str,
        tool_def,
        tool_timeout: int = 30,
    ) -> None:
        self._pool = pool
        self._server_name = server_name
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        for attempt in range(2):
            try:
                acting_as = _acting_as_ctx.get()
                session = await self._pool.get_session(self._server_name, acting_as)
                result = await asyncio.wait_for(
                    session.call_tool(self._original_name, arguments=kwargs),
                    timeout=self._tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP tool '{}' timed out after {}s", self._name, self._tool_timeout
                )
                return f"(MCP tool call timed out after {self._tool_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
                return "(MCP tool call was cancelled)"
            except Exception as exc:
                if _is_transient(exc):
                    if attempt == 0:
                        logger.warning(
                            "MCP tool '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.error(
                        "MCP tool '{}' failed after retry: {}: {}",
                        self._name,
                        type(exc).__name__,
                        exc,
                    )
                    return f"(MCP tool call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP tool '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP tool call failed: {type(exc).__name__})"
            else:
                parts = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP tool call failed)"


class MCPResourceWrapper(Tool):
    def __init__(
        self,
        pool: MCPClientPool,
        server_name: str,
        resource_def,
        resource_timeout: int = 30,
    ) -> None:
        self._pool = pool
        self._server_name = server_name
        self._uri = resource_def.uri
        self._name = f"mcp_{server_name}_resource_{resource_def.name}"
        desc = resource_def.description or resource_def.name
        self._description = f"[MCP Resource] {desc}\nURI: {self._uri}"
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._resource_timeout = resource_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        for attempt in range(2):
            try:
                acting_as = _acting_as_ctx.get()
                session = await self._pool.get_session(self._server_name, acting_as)
                result = await asyncio.wait_for(
                    session.read_resource(self._uri),
                    timeout=self._resource_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP resource '{}' timed out after {}s",
                    self._name,
                    self._resource_timeout,
                )
                return f"(MCP resource read timed out after {self._resource_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP resource '{}' was cancelled by server/SDK", self._name)
                return "(MCP resource read was cancelled)"
            except Exception as exc:
                if _is_transient(exc):
                    if attempt == 0:
                        logger.warning(
                            "MCP resource '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.error(
                        "MCP resource '{}' failed after retry: {}: {}",
                        self._name,
                        type(exc).__name__,
                        exc,
                    )
                    return f"(MCP resource read failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP resource '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP resource read failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for block in result.contents:
                    if isinstance(block, types.TextResourceContents):
                        parts.append(block.text)
                    elif isinstance(block, types.BlobResourceContents):
                        parts.append(f"[Binary resource: {len(block.blob)} bytes]")
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP resource read failed)"


class MCPPromptWrapper(Tool):
    def __init__(
        self,
        pool: MCPClientPool,
        server_name: str,
        prompt_def,
        prompt_timeout: int = 30,
    ) -> None:
        self._pool = pool
        self._server_name = server_name
        self._prompt_name = prompt_def.name
        self._name = f"mcp_{server_name}_prompt_{prompt_def.name}"
        desc = prompt_def.description or prompt_def.name
        self._description = (
            f"[MCP Prompt] {desc}\n"
            "Returns a filled prompt template that can be used as a workflow guide."
        )
        self._prompt_timeout = prompt_timeout

        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in prompt_def.arguments or []:
            prop: dict[str, Any] = {"type": "string"}
            if getattr(arg, "description", None):
                prop["description"] = arg.description
            properties[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        from mcp.shared.exceptions import McpError

        for attempt in range(2):
            try:
                acting_as = _acting_as_ctx.get()
                session = await self._pool.get_session(self._server_name, acting_as)
                result = await asyncio.wait_for(
                    session.get_prompt(self._prompt_name, arguments=kwargs),
                    timeout=self._prompt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP prompt '{}' timed out after {}s",
                    self._name,
                    self._prompt_timeout,
                )
                return f"(MCP prompt call timed out after {self._prompt_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP prompt '{}' was cancelled by server/SDK", self._name)
                return "(MCP prompt call was cancelled)"
            except McpError as exc:
                logger.error(
                    "MCP prompt '{}' failed: code={} message={}",
                    self._name,
                    exc.error.code,
                    exc.error.message,
                )
                return f"(MCP prompt call failed: {exc.error.message} [code {exc.error.code}])"
            except Exception as exc:
                if _is_transient(exc):
                    if attempt == 0:
                        logger.warning(
                            "MCP prompt '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.error(
                        "MCP prompt '{}' failed after retry: {}: {}",
                        self._name,
                        type(exc).__name__,
                        exc,
                    )
                    return f"(MCP prompt call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP prompt '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP prompt call failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for message in result.messages:
                    content = message.content
                    if isinstance(content, types.TextContent):
                        parts.append(content.text)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, types.TextContent):
                                parts.append(block.text)
                            else:
                                parts.append(str(block))
                    else:
                        parts.append(str(content))
                return "\n".join(parts) or "(no output)"

        return "(MCP prompt call failed)"


# ---------------------------------------------------------------------------
# connect_mcp_servers — discovery + registry, returns the live pool
# ---------------------------------------------------------------------------


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry
) -> tuple[MCPClientPool, int]:
    """Connect to each configured MCP server (one anonymous-identity session
    per server) for tool/resource/prompt schema discovery, registering
    pool-backed wrappers in the registry. Per-identity transports are opened
    lazily on first execute() with that acting_as.

    Returns (pool, succeeded_count). succeeded_count == 0 means no server's
    discovery completed; callers should treat that as "not connected" and
    retry on the next dispatch so a transiently-down MCP can be picked up.
    """
    pool = MCPClientPool(mcp_servers)
    succeeded = 0

    async def discover(name: str, cfg) -> bool:
        try:
            session = await pool.get_session(name, None)
        except Exception as e:
            hint = ""
            text = str(e).lower()
            if any(
                marker in text
                for marker in (
                    "parse error",
                    "invalid json",
                    "unexpected token",
                    "jsonrpc",
                    "content-length",
                )
            ):
                hint = (
                    " Hint: this looks like stdio protocol pollution. Make sure the MCP server writes "
                    "only JSON-RPC to stdout and sends logs/debug output to stderr instead."
                )
            logger.error("MCP server '{}': failed to connect: {}{}", name, e, hint)
            return False

        try:
            tools = await session.list_tools()
        except Exception as e:
            logger.error("MCP server '{}': list_tools failed: {}", name, e)
            return False

        enabled_tools = set(cfg.enabled_tools)
        allow_all_tools = "*" in enabled_tools
        registered_count = 0
        matched_enabled_tools: set[str] = set()
        available_raw_names = [tool_def.name for tool_def in tools.tools]
        available_wrapped_names = [f"mcp_{name}_{tool_def.name}" for tool_def in tools.tools]

        for tool_def in tools.tools:
            wrapped_name = f"mcp_{name}_{tool_def.name}"
            if (
                not allow_all_tools
                and tool_def.name not in enabled_tools
                and wrapped_name not in enabled_tools
            ):
                logger.debug(
                    "MCP: skipping tool '{}' from server '{}' (not in enabledTools)",
                    wrapped_name,
                    name,
                )
                continue
            wrapper = MCPToolWrapper(pool, name, tool_def, tool_timeout=cfg.tool_timeout)
            registry.register(wrapper)
            logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)
            registered_count += 1
            if enabled_tools:
                if tool_def.name in enabled_tools:
                    matched_enabled_tools.add(tool_def.name)
                if wrapped_name in enabled_tools:
                    matched_enabled_tools.add(wrapped_name)

        if enabled_tools and not allow_all_tools:
            unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
            if unmatched_enabled_tools:
                logger.warning(
                    "MCP server '{}': enabledTools entries not found: {}. "
                    "Available raw names: {}. Available wrapped names: {}",
                    name,
                    ", ".join(unmatched_enabled_tools),
                    ", ".join(available_raw_names) or "(none)",
                    ", ".join(available_wrapped_names) or "(none)",
                )

        try:
            resources_result = await session.list_resources()
            for resource in resources_result.resources:
                wrapper = MCPResourceWrapper(
                    pool, name, resource, resource_timeout=cfg.tool_timeout
                )
                registry.register(wrapper)
                registered_count += 1
                logger.debug(
                    "MCP: registered resource '{}' from server '{}'", wrapper.name, name
                )
        except Exception as e:
            logger.debug("MCP server '{}': resources not supported or failed: {}", name, e)

        try:
            prompts_result = await session.list_prompts()
            for prompt in prompts_result.prompts:
                wrapper = MCPPromptWrapper(
                    pool, name, prompt, prompt_timeout=cfg.tool_timeout
                )
                registry.register(wrapper)
                registered_count += 1
                logger.debug("MCP: registered prompt '{}' from server '{}'", wrapper.name, name)
        except Exception as e:
            logger.debug("MCP server '{}': prompts not supported or failed: {}", name, e)

        logger.info(
            "MCP server '{}': connected, {} capabilities registered",
            name,
            registered_count,
        )
        return True

    # Sequentialise discovery so every transport stack is entered in the
    # caller's task; any subsequent close() from the same task can then exit
    # those stacks without anyio's "different task" cancel-scope mismatch.
    for name, cfg in mcp_servers.items():
        try:
            if await discover(name, cfg):
                succeeded += 1
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            logger.error("MCP server '{}' discovery failed: {}", name, e)

    return pool, succeeded

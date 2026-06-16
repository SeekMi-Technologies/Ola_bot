"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import json as _json
import time
import uuid
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.agent.admin_context import get_acting_admin_id
from nanobot.agent.tools.mcp import set_acting_as
from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import provision_admin, safe_filename
from nanobot.utils.media_decode import (
    FileSizeExceeded as _FileSizeExceeded,
    MAX_FILE_SIZE,
    save_base64_data_url as _save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

__all__ = (
    "MAX_FILE_SIZE",
    "_FileSizeExceeded",
    "_save_base64_data_url",
    "create_app",
    "handle_chat_completions",
)


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(
    content: str,
    model: str,
    *,
    metadata: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # When usage is supplied (Ola CRM #98 — auto-title path uses non-streaming
    # to track token spend), pass it through verbatim. Otherwise fall back to
    # the zero placeholder so OpenAI clients always see the field.
    usage_payload: dict[str, Any] = (
        dict(usage) if isinstance(usage, dict) and usage
        else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    resp: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage_payload,
    }
    if metadata:
        resp["metadata"] = metadata
    return resp


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload)}\n\n".encode()


def _sse_tool_event(event: dict[str, Any]) -> bytes:
    """Format a single tool_event SSE frame as a named event.

    Standard OpenAI SSE clients ignore named events (they only consume
    default-event chat.completion.chunk frames), so adding this is
    backwards-compatible. Consumers that subscribe to `tool_event` (e.g.
    Ola CRM olaController) get real-time tool start/end notifications.
    """
    return f"event: tool_event\ndata: {_json.dumps(event)}\n\n".encode()


def _sse_usage(usage: dict[str, Any]) -> bytes:
    """Format a single usage SSE frame as a named event (Ola CRM #98).

    Same backwards-compatible mechanism as tool_event — standard OpenAI
    clients ignore named events. Schema is the 7-field flat dict produced
    by AgentLoop._last_usage (loop.py): provider, model, prompt_tokens,
    completion_tokens, total_tokens, cached_tokens, iterations. Consumed
    by Ola CRM olaController/chat.js to write LlmUsage rows.
    """
    return f"event: usage\ndata: {_json.dumps(usage)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _parse_json_content(body: dict) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: web.Request) -> tuple[str, list[str], str | None, str | None]:
    """Parse multipart/form-data. Returns (text, media_paths, session_id, model)."""
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    media_paths: list[str] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — supports JSON and multipart/form-data."""
    # Extract acting-as identity from X-Ola-Acting-As and store in contextvar
    # before any await. MCPToolWrapper.execute reads it in the caller's task
    # to pick the matching transport from MCPClientPool, which has the right
    # X-Acting-As baked into the httpx client headers. Header absent → stays
    # None → backend falls back to systemAdmin.
    set_acting_as(request.headers.get("X-Ola-Acting-As"))
    provision_admin(request.app["agent_loop"].workspace, get_acting_admin_id())

    content_type = request.content_type or ""
    if not isinstance(content_type, str):
        content_type = ""

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "nanobot")

    stream = False
    try:
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model = await _parse_multipart(request)
        else:
            try:
                body = await request.json()
            except Exception:
                return _error_json(400, "Invalid JSON body")
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    # -- streaming path --
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        resp.enable_compression()
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        # Queue items:
        #   ("text", str)         streamed text delta from the LLM
        #   ("tool_event", dict)  tool start/end progress event
        #   ("usage", dict)       per-turn token telemetry (Ola CRM #98), pushed
        #                         once from _run() finally block AFTER
        #                         process_direct sets agent_loop._last_usage
        #   ("end", None)         consumer-loop terminator, also pushed from
        #                         _run() finally so it always lands AFTER usage
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stream_failed = False

        async def _on_stream(token: str) -> None:
            await queue.put(("text", token))

        async def _on_stream_end(*_a: Any, resuming: bool = False, **_kw: Any) -> None:
            # resuming=True means the agent is pausing to run tools; more
            # streamed text will follow. resuming=False = real end of stream.
            # No-op here: end + usage are both pushed from _run()'s finally
            # block so we can include the freshly-set _last_usage in-band.
            return

        # Real-time tool_event progress callback. Mirrors _capture_tool_events()
        # in the non-stream path (which batches into metadata.tool_events at the
        # end), but streams each event into the SSE response as it happens.
        # Signature must accept (content, *, tool_hint, tool_events) per
        # nanobot/utils/progress_events.py:invoke_on_progress contract.
        async def _on_progress(
            content: str,
            *,
            tool_hint: bool = False,
            tool_events: list[dict[str, Any]] | None = None,
        ) -> None:
            if not tool_events:
                return
            for ev in tool_events:
                await queue.put(("tool_event", ev))

        async def _run() -> None:
            nonlocal stream_failed
            try:
                async with session_lock:
                    await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                            on_progress=_on_progress,
                        ),
                        timeout=timeout_s,
                    )
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
            finally:
                # Emit usage telemetry and the end signal as the last two queue
                # items. _last_usage is set inside process_direct AFTER the
                # runner returns, so it must be read here (after the await),
                # not from inside _on_stream_end. Errored turns may still have
                # partial usage data — we emit it so the dashboard can record
                # spend on failed turns (Ola CRM #98). isinstance guard guards
                # against MagicMock'd agent_loops in test fixtures and any
                # malformed _last_usage state.
                last_usage = getattr(agent_loop, "_last_usage", None)
                if isinstance(last_usage, dict) and last_usage:
                    await queue.put(("usage", last_usage))
                await queue.put(("end", None))

        task = asyncio.create_task(_run())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    break
                if kind == "text":
                    await resp.write(_sse_chunk(payload, model_name, chunk_id))
                elif kind == "tool_event":
                    await resp.write(_sse_tool_event(payload))
                elif kind == "usage":
                    await resp.write(_sse_usage(payload))
        finally:
            task.cancel()

        if not stream_failed:
            await resp.write(_sse_chunk("", model_name, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
    _FALLBACK = EMPTY_FINAL_RESPONSE_MESSAGE

    # Collect structured tool events emitted by the agent loop. Mirrors what
    # bus channels do (see _bus_progress in agent/loop.py): each tool call
    # publishes a "start" payload and a "end"/"error" payload. We surface
    # them in the response so HTTP clients (e.g. CRM olaController) can render
    # rich blocks without having to subscribe to the bus.
    captured_tool_events: list[dict[str, Any]] = []

    async def _capture_tool_events(
        content: str,
        *,
        tool_hint: bool = False,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> None:
        if tool_events:
            captured_tool_events.extend(tool_events)

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                        on_progress=_capture_tool_events,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, retrying", session_key)
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_progress=_capture_tool_events,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning("Empty response after retry, using fallback")
                        response_text = _FALLBACK

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    extra_metadata = {"tool_events": captured_tool_events} if captured_tool_events else None
    last_usage = getattr(agent_loop, "_last_usage", None)
    usage_payload = last_usage if isinstance(last_usage, dict) and last_usage else None
    return web.json_response(
        _chat_completion_response(
            response_text,
            model_name,
            metadata=extra_metadata,
            usage=usage_payload,
        )
    )


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = request.app.get("model_name", "nanobot")
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop, model_name: str = "nanobot", request_timeout: float = 120.0
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    app = web.Application(client_max_size=20 * 1024 * 1024)  # 20MB for base64 images
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app

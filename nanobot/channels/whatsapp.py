"""WhatsApp channel implementation using Node.js bridge.

Multi-tenant mode: ChannelManager file-system-scans ~/.nanobot/wa/<adminId>/auth/
and constructs one WhatsAppChannel per discovered admin, each bound to the
admin_id at construction. Inbound metadata carries `_acting_as=<adminId>` so
the agent loop and MCP client pool isolate per tenant (replicates AskOla §2
5-layer chain in doc/multitenancy_current_state.md).

Per-admin token = HMAC-SHA256(MCP_SERVICE_TOKEN, adminId) — derived identically
by bridge (Node) and this channel (Python) with no shared state.
"""

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class WhatsAppConfig(Base):
    """WhatsApp channel configuration."""

    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"  # multi-tenant 模式下由 ChannelManager 派生覆盖
    bridge_token: str = ""  # legacy single-tenant; multi-tenant 用 HMAC 派生
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "open"  # "open" responds to all, "mention" only when @mentioned
    admin_id: str = ""  # CRM Admin._id (24-hex). 非空 → multi-tenant 模式 (注 _acting_as + 用 HMAC token)
    # Voice transcription — falls back to env vars if empty
    transcription_provider: str = "groq"
    transcription_api_key: str = ""
    transcription_api_base: str = ""
    transcription_language: str | None = None
    # CRM audio upload — upload inbound WhatsApp audio to CRM File storage
    # so the agent can use file.transcribe / file.get_transcript MCP tools.
    # Format: http://<host>:<port> (no trailing slash). Empty = disabled.
    crm_upload_url: str = ""
    crm_service_token: str = ""  # Bearer token; defaults to MCP_SERVICE_TOKEN env


def _bridge_token_path() -> Path:
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("whatsapp-auth") / "bridge-token"


def _load_or_create_bridge_token(path: Path) -> str:
    """Load a persisted bridge token or create one on first use."""
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel that connects to a Node.js bridge.

    The bridge uses @whiskeysockets/baileys to handle the WhatsApp Web protocol.
    Communication between Python and Node.js is via WebSocket.
    """

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WhatsAppConfig.model_validate(config)
        super().__init__(config, bus)
        self._ws = None
        self._connected = False
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        # In-memory LID→PN cache: per-WhatsAppChannel instance (天然 per-admin
        # 隔离 in multi-tenant mode). Cross-restart 持久化 → handoff H8.
        self._lid_to_phone: dict[str, str] = {}
        self._bridge_token: str | None = None
        self._admin_id: str = config.admin_id  # 实例级绑定 (空 = legacy single-tenant)
        # Propagate transcription config from channel config → BaseChannel defaults;
        # fall back to env vars (GROQ_API_KEY / OPENAI_API_KEY) when config is empty.
        self.transcription_provider = config.transcription_provider or "groq"
        self.transcription_api_key = (
            config.transcription_api_key
            or os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.transcription_api_base = config.transcription_api_base or ""
        self.transcription_language = config.transcription_language

    def _effective_bridge_token(self) -> str:
        """Resolve the bridge token.

        Multi-tenant (admin_id set): derive HMAC-SHA256(MCP_SERVICE_TOKEN, adminId).
        Same algorithm as bridge/src/server.ts tokenFor() — both sides compute
        identical bytes without any shared table.

        Legacy single-tenant: load/create local bridge-token file (old behavior).
        """
        if self._bridge_token is not None:
            return self._bridge_token

        if self._admin_id:
            secret = os.environ.get("MCP_SERVICE_TOKEN", "").strip()
            if not secret:
                raise RuntimeError(
                    "MCP_SERVICE_TOKEN env required for multi-tenant WhatsApp channel"
                )
            self._bridge_token = hmac.new(
                secret.encode("utf-8"),
                self._admin_id.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return self._bridge_token

        configured = self.config.bridge_token.strip()
        if configured:
            self._bridge_token = configured
        else:
            self._bridge_token = _load_or_create_bridge_token(_bridge_token_path())
        return self._bridge_token

    async def login(self, force: bool = False) -> bool:
        """
        Set up and run the WhatsApp bridge for QR code login.

        This spawns the Node.js bridge process which handles the WhatsApp
        authentication flow. The process blocks until the user scans the QR code
        or interrupts with Ctrl+C.
        """
        try:
            bridge_dir = _ensure_bridge_setup()
        except RuntimeError as e:
            logger.error("{}", e)
            return False

        env = {**os.environ}
        # Bridge now reads MCP_SERVICE_TOKEN + AUTH_ROOT (multi-tenant mode);
        # the old BRIDGE_TOKEN/AUTH_DIR pair was removed in the multi-tenant
        # rewrite. login() spawns the bridge as a child for one-off QR pairing,
        # so it just needs the canonical env propagated.
        if "MCP_SERVICE_TOKEN" not in env or not env["MCP_SERVICE_TOKEN"].strip():
            logger.error(
                "MCP_SERVICE_TOKEN required for WhatsApp bridge login; "
                "ensure it is set in the calling shell (start-dev.sh + .secrets/SERVERS.env)."
            )
            return False
        env["AUTH_ROOT"] = str(Path.home() / ".nanobot" / "wa")
        if self._admin_id:
            # Run the spawned bridge restricted to this admin (parity with how
            # start-dev launches it in single-shared-bridge mode without
            # SINGLE_ADMIN_ID; login is typically per-admin so restrict here).
            env["SINGLE_ADMIN_ID"] = self._admin_id

        logger.info("Starting WhatsApp bridge for QR login...")
        try:
            subprocess.run(
                [shutil.which("npm"), "start"], cwd=bridge_dir, check=True, env=env
            )
        except subprocess.CalledProcessError:
            return False

        return True

    def _resolve_ws_url(self) -> str:
        """Build the WebSocket URL for the next connect attempt.

        Multi-tenant mode: re-read the portfile every iteration so the existing
        reconnect loop picks up bridge restarts (port changes) without needing the
        registry to recreate the channel. Two portfile locations checked in order:

          1. Per-admin: ~/.nanobot/wa/<adminId>/port  (multi-bridge mode — each
             admin has a dedicated bridge process started with SINGLE_ADMIN_ID env)
          2. Shared:   ~/.nanobot/wa/bridge.port      (single shared bridge serving
             N admins via URL routing)

        Falls back to config.bridge_url if neither portfile is usable.

        Legacy single-tenant mode (admin_id empty): return static config.bridge_url.
        """
        if not self._admin_id:
            return self.config.bridge_url

        base_url = self.config.bridge_url
        wa_root = Path.home() / ".nanobot" / "wa"
        for portfile in (wa_root / self._admin_id / "port", wa_root / "bridge.port"):
            if portfile.exists():
                try:
                    port = int(portfile.read_text().strip())
                    if port > 0:
                        base_url = f"ws://127.0.0.1:{port}"
                        break
                except (ValueError, OSError):
                    continue

        token = self._effective_bridge_token()
        return f"{base_url}/wa/{self._admin_id}?token={token}"

    async def start(self) -> None:
        """Start the WhatsApp channel by connecting to the bridge.

        Multi-tenant mode (admin_id set): connect to ws://<host>:<port>/wa/<adminId>?token=<hmac>;
          token verified by bridge during WS upgrade, so no auth message needed.
          ws_url re-derived per iteration → survives bridge restarts.
        Legacy single-tenant mode (admin_id empty): connect to bare base URL,
          send {type:auth,token:...} as first message (old behavior).
        """
        import websockets

        log_tag = f"[{self._admin_id}]" if self._admin_id else ""
        self._running = True

        while self._running:
            ws_url = self._resolve_ws_url()  # fresh per iteration
            logger.info("{} Connecting to WhatsApp bridge at {}...", log_tag, ws_url)

            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    if not self._admin_id:
                        # Legacy: auth via first message (multi-tenant uses URL ?token=)
                        await ws.send(
                            json.dumps({"type": "auth", "token": self._effective_bridge_token()})
                        )
                    self._connected = True
                    logger.info("{} Connected to WhatsApp bridge", log_tag)

                    # Listen for messages
                    async for message in ws:
                        try:
                            await self._handle_bridge_message(message)
                        except Exception as e:
                            logger.error("{} Error handling bridge message: {}", log_tag, e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                logger.warning("{} WhatsApp bridge connection error: {}", log_tag, e)

                if self._running:
                    logger.info("{} Reconnecting in 5 seconds...", log_tag)
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the WhatsApp channel."""
        self._running = False
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through WhatsApp."""
        if not self._ws or not self._connected:
            logger.warning("WhatsApp bridge not connected")
            return

        chat_id = msg.chat_id

        if msg.content:
            try:
                payload = {"type": "send", "to": chat_id, "text": msg.content}
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                logger.error("Error sending WhatsApp message: {}", e)
                raise

        for media_path in msg.media or []:
            try:
                mime, _ = mimetypes.guess_type(media_path)
                payload = {
                    "type": "send_media",
                    "to": chat_id,
                    "filePath": media_path,
                    "mimetype": mime or "application/octet-stream",
                    "fileName": media_path.rsplit("/", 1)[-1],
                }
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                logger.error("Error sending WhatsApp media {}: {}", media_path, e)
                raise

    async def _upload_audio_to_crm(self, file_path: str) -> dict | None:
        """Upload an audio file to CRM via the internal upload-audio endpoint.

        Returns the JSON response body on success (contains fileId, transcriptionJobId),
        or None on any error (logged, non-blocking).
        """
        url = self.config.crm_upload_url
        if not url:
            return None
        token = self.config.crm_service_token or os.environ.get("MCP_SERVICE_TOKEN", "")
        if not token:
            logger.debug("[crm-upload] skipped: no service token")
            return None

        # admin_id may be empty in single-tenant dev mode.
        # The server will fall back to system admin when X-Acting-As is absent.
        admin_id = self._admin_id

        try:
            import httpx

            p = Path(file_path)
            if not p.exists():
                logger.warning("[crm-upload] file not found: {}", file_path)
                return None

            mime = mimetypes.guess_type(str(p))[0] or "audio/ogg"
            endpoint = f"{url}/internal/upload-audio"
            headers = {"Authorization": f"Bearer {token}"}
            if admin_id:
                headers["X-Acting-As"] = admin_id
            async with httpx.AsyncClient(timeout=30) as client:
                with open(p, "rb") as f:
                    resp = await client.post(
                        endpoint,
                        files={"file": (p.name, f, mime)},
                        headers=headers,
                    )
                if resp.status_code >= 400:
                    logger.warning("[crm-upload] {} returned {}: {}", endpoint, resp.status_code, resp.text[:200])
                    return None
                data = resp.json()
                logger.info("[crm-upload] uploaded {} → fileId={} deduped={}", p.name, data.get("fileId"), data.get("deduped"))
                return data
        except Exception as e:
            logger.warning("[crm-upload] failed for {}: {}", file_path, e)
            return None

    async def _handle_bridge_message(self, raw: str) -> None:
        """Handle a message from the bridge."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from bridge: {}", raw[:100])
            return

        msg_type = data.get("type")

        if msg_type == "message":
            # Incoming message from WhatsApp
            # Deprecated by whatsapp: old phone number style typically: <phone>@s.whatspp.net
            pn = data.get("pn", "")
            # New LID sytle typically:
            sender = data.get("sender", "")
            content = data.get("content", "")
            message_id = data.get("id", "")

            if message_id:
                if message_id in self._processed_message_ids:
                    return
                self._processed_message_ids[message_id] = None
                while len(self._processed_message_ids) > 1000:
                    self._processed_message_ids.popitem(last=False)

            # Extract just the phone number or lid as chat_id
            is_group = data.get("isGroup", False)

            # P0: 完全不接群消息 (Baileys group bugs: #1505/#1935/#2233);
            # group_policy 'mention' / 白名单策略 → handoff (re-enable by expanding
            # the Literal enum and restoring the mention-based logic).
            if is_group:
                return

            # Classify by JID suffix: @s.whatsapp.net = phone, @lid.whatsapp.net = LID
            # The bridge's pn/sender fields don't consistently map to phone/LID across versions.
            raw_a = pn or ""
            raw_b = sender or ""
            id_a = raw_a.split("@")[0] if "@" in raw_a else raw_a
            id_b = raw_b.split("@")[0] if "@" in raw_b else raw_b

            phone_id = ""
            lid_id = ""
            for raw, extracted in [(raw_a, id_a), (raw_b, id_b)]:
                if "@s.whatsapp.net" in raw:
                    phone_id = extracted
                elif "@lid.whatsapp.net" in raw:
                    lid_id = extracted
                elif extracted and not phone_id:
                    phone_id = extracted  # best guess for bare values

            # In-memory LID→PN cache (per-instance, dies on restart). Persistence
            # 跨重启 → handoff H8 (Baileys #2263 lid-mapping.update 仍不可靠时,
            # 用 messages.upsert dual-ID 双向绑 + write to state.json/Mongo).
            if phone_id and lid_id:
                self._lid_to_phone[lid_id] = phone_id
            sender_id = phone_id or self._lid_to_phone.get(lid_id, "") or lid_id or id_a or id_b

            _tag = f"[{self._admin_id}]" if self._admin_id else ""
            logger.info("{} Sender phone={} lid={} → sender_id={}", _tag, phone_id or "(empty)", lid_id or "(empty)", sender_id)

            # Extract media paths (images/documents/videos downloaded by the bridge)
            media_paths = data.get("media") or []

            # Handle voice transcription if it's a voice message
            voice_transcribed = False
            if content == "[Voice Message]":
                if media_paths:
                    # Upload to CRM first (await so agent knows fileId)
                    crm_result = await self._upload_audio_to_crm(media_paths[0])
                    crm_file_tag = ""
                    if crm_result and crm_result.get("fileId"):
                        crm_file_tag = f"\n[CRM文件已上传 fileId={crm_result['fileId']}]"
                    logger.info("Transcribing voice message from {}...", sender_id)
                    transcription = await self.transcribe_audio(media_paths[0])
                    if transcription:
                        content = f"[语音消息转写] {transcription}{crm_file_tag}"
                        voice_transcribed = True
                        logger.info("Transcribed voice from {}: {}...", sender_id, transcription[:50])
                    else:
                        content = "[Voice Message: Transcription failed]"
                else:
                    content = "[Voice Message: Audio not available]"

            # Build content tags matching Telegram's pattern: [image: /path] or [file: /path]
            # Skip audio file tag when transcription succeeded (text already contains transcript)
            remaining_media = []
            for p in media_paths:
                if voice_transcribed:
                    # Drop the transcribed audio file from media — transcript is the content
                    continue
                mime, _ = mimetypes.guess_type(p)

                # Auto-transcribe audio document attachments (.wav, .mp3, .ogg, .m4a, etc.)
                if mime and mime.startswith("audio/"):
                    # Upload to CRM first (await so agent knows fileId)
                    crm_result = await self._upload_audio_to_crm(p)
                    logger.info("Transcribing audio attachment {}...", p)
                    transcription = await self.transcribe_audio(p)
                    if transcription:
                        tag = f"[音频文件转写] {transcription}"
                        if crm_result and crm_result.get("fileId"):
                            tag += f"\n[CRM文件已上传 fileId={crm_result['fileId']}]"
                        content = f"{content}\n{tag}" if content else tag
                        logger.info("Transcribed audio attachment: {}...", transcription[:50])
                        continue  # Don't add [file: /path] tag
                    # Transcription failed — fall through to regular file tag

                media_type = "image" if mime and mime.startswith("image/") else "file"
                media_tag = f"[{media_type}: {p}]"
                content = f"{content}\n{media_tag}" if content else media_tag
                remaining_media.append(p)

            metadata: dict[str, Any] = {
                "message_id": message_id,
                "timestamp": data.get("timestamp"),
                "is_group": False,  # group already dropped above
            }
            if self._admin_id:
                # ★ Multi-tenant isolation key — agent loop reads this from metadata,
                # sets the ContextVar, and the MCP client pool injects X-Acting-As
                # on outgoing tool calls. Mirrors email.py:191 pattern. The same
                # adminId is enforced upstream by bridge URL routing, so this can
                # be trusted as the source of truth for the channel's tenant scope.
                metadata["_acting_as"] = self._admin_id

            await self._handle_message(
                sender_id=sender_id,
                chat_id=sender,  # Use full LID for replies
                content=content,
                media=remaining_media if voice_transcribed else media_paths,
                metadata=metadata,
            )

        elif msg_type == "status":
            # Connection status update
            status = data.get("status")
            logger.info("WhatsApp status: {}", status)

            if status == "connected":
                self._connected = True
            elif status == "disconnected":
                self._connected = False

        elif msg_type == "qr":
            # QR code for authentication
            logger.info("Scan QR code in the bridge terminal to connect WhatsApp")

        elif msg_type == "error":
            logger.error("WhatsApp bridge error: {}", data.get("error"))


def _ensure_bridge_setup() -> Path:
    """
    Ensure the WhatsApp bridge is set up and built.

    Returns the bridge directory. Raises RuntimeError if npm is not found
    or bridge cannot be built.
    """
    from nanobot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()

    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    npm_path = shutil.which("npm")
    if not npm_path:
        raise RuntimeError("npm not found. Please install Node.js >= 18.")

    # Find source bridge
    current_file = Path(__file__)
    pkg_bridge = current_file.parent.parent / "bridge"
    src_bridge = current_file.parent.parent.parent / "bridge"

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        raise RuntimeError(
            "WhatsApp bridge source not found. "
            "Try reinstalling: pip install --force-reinstall nanobot"
        )

    logger.info("Setting up WhatsApp bridge...")
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    logger.info("  Installing dependencies...")
    subprocess.run([npm_path, "install"], cwd=user_bridge, check=True, capture_output=True)

    logger.info("  Building...")
    subprocess.run([npm_path, "run", "build"], cwd=user_bridge, check=True, capture_output=True)

    logger.info("Bridge ready")
    return user_bridge

"""WhatsApp multi-tenant registry: file-system discovery + 30s poll.

In multi-tenant mode, the single 'whatsapp' channel created from
config.channels.whatsapp is replaced by N 'whatsapp:<adminId>' instances,
one per directory matching ~/.nanobot/wa/<adminId>/auth/. A 30s background
poll loop adds/removes channels as the operator mkdir/rm-rf those dirs.

The base WhatsAppConfig (allow_from, group_policy, transcription settings)
is snapshotted from the original config so each per-admin instance inherits
the same operational policy.

See doc/whatsapp_baileys_multitenant.md and the plan file for the rationale
behind file-system-driven discovery (vs MCP query / Mongo change streams).
"""

from __future__ import annotations

import asyncio
import glob
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.channels.whatsapp import WhatsAppChannel, WhatsAppConfig

if TYPE_CHECKING:
    from nanobot.channels.manager import ChannelManager


WA_ROOT = Path("~/.nanobot/wa").expanduser()
PORTFILE = WA_ROOT / "bridge.port"
_ADMIN_ID_HEX_LEN = 24  # CRM Admin._id is 24-char ObjectId hex
_HEX_CHARS = set("0123456789abcdef")


def _read_bridge_port() -> int | None:
    if not PORTFILE.exists():
        return None
    try:
        n = int(PORTFILE.read_text().strip())
        return n if n > 0 else None
    except (ValueError, OSError):
        return None


def _scan_admin_dirs() -> set[str]:
    """Return {admin_id} for every directory matching ~/.nanobot/wa/<adminId>/auth/."""
    out: set[str] = set()
    for auth_dir in glob.glob(str(WA_ROOT / "*" / "auth")):
        admin_id = Path(auth_dir).parent.name
        if len(admin_id) != _ADMIN_ID_HEX_LEN or any(c not in _HEX_CHARS for c in admin_id):
            logger.warning("Skipping non-ObjectId WA directory: {}", admin_id)
            continue
        out.add(admin_id)
    return out


class WhatsAppMultiTenantRegistry:
    """Lifecycle owner for per-admin WhatsAppChannel instances.

    The initial expand() runs at ChannelManager.__init__ time (synchronous),
    swapping the single placeholder for N per-admin channels. tick() runs every
    30s from wa_poll_loop() to add/remove channels as the operator changes
    ~/.nanobot/wa/.

    base_config is None when no WhatsApp channel was originally configured,
    in which case the registry is inert (poll loop yields immediately).
    """

    def __init__(self, manager: "ChannelManager"):
        self.manager = manager
        original = manager.channels.get("whatsapp")
        if not isinstance(original, WhatsAppChannel):
            self.base_config: WhatsAppConfig | None = None
            self._transcription: dict[str, str] = {}
            return
        # Snapshot config + transcription settings from the placeholder
        self.base_config = WhatsAppConfig.model_validate(original.config.model_dump())
        self._transcription = {
            "transcription_provider": getattr(original, "transcription_provider", ""),
            "transcription_api_key": getattr(original, "transcription_api_key", ""),
            "transcription_api_base": getattr(original, "transcription_api_base", ""),
            "transcription_language": getattr(original, "transcription_language", ""),
        }

    def expand(self) -> None:
        """Initial scan: build channels for each admin dir, drop the placeholder.

        Bridge readiness is no longer required at expand time — each channel's
        WhatsAppChannel._resolve_ws_url re-reads the per-admin OR shared portfile
        on every reconnect attempt, so channels created here will keep trying
        until bridge(s) come up. This supports both single-shared-bridge and
        multi-bridge (one bridge per admin) deployment modes uniformly.

        If no admin dirs exist on disk yet, the placeholder is kept; the poll
        loop adds channels as operators mkdir new admin dirs.
        """
        if self.base_config is None:
            return

        admin_ids = _scan_admin_dirs()
        if not admin_ids:
            logger.info(
                "WhatsApp multi-tenant: no admin dirs under {} yet; poll loop will pick up additions.",
                WA_ROOT,
            )
            return

        for admin_id in sorted(admin_ids):
            self._add_channel(admin_id)

        # The placeholder never had admin_id and never started; drop it now that
        # per-admin instances exist.
        self.manager.channels.pop("whatsapp", None)
        logger.info("WhatsApp multi-tenant init: {} admin channel(s) registered", len(admin_ids))

    def _add_channel(self, admin_id: str, port: int | None = None) -> WhatsAppChannel:
        """Build a per-admin channel. `port` arg is now optional; when omitted
        the channel uses its default bridge_url placeholder and _resolve_ws_url
        will dynamically read the portfile on each connect attempt.
        """
        update: dict[str, str] = {"admin_id": admin_id}
        if port is not None:
            update["bridge_url"] = f"ws://127.0.0.1:{port}"
        cfg = self.base_config.model_copy(update=update)
        ch = WhatsAppChannel(cfg, self.manager.bus)
        for attr, val in self._transcription.items():
            setattr(ch, attr, val)
        self.manager.channels[f"whatsapp:{admin_id}"] = ch
        return ch

    async def tick(self) -> None:
        """One poll iteration: diff disk vs current channels; add/remove.

        Like expand(), no longer requires bridge readiness — the channel's own
        reconnect loop handles bridge availability.
        """
        if self.base_config is None:
            return

        on_disk = _scan_admin_dirs()
        current = {
            k.removeprefix("whatsapp:")
            for k in list(self.manager.channels.keys())
            if k.startswith("whatsapp:")
        }

        for admin_id in on_disk - current:
            logger.info("WA poll: discovered new admin {}", admin_id)
            ch = self._add_channel(admin_id)
            asyncio.create_task(ch.start())

        for admin_id in current - on_disk:
            logger.info("WA poll: admin removed {}", admin_id)
            ch = self.manager.channels.pop(f"whatsapp:{admin_id}", None)
            if ch:
                try:
                    await ch.stop()
                except Exception as e:
                    logger.warning("Error stopping WA channel {}: {}", admin_id, e)


async def wa_poll_loop(
    registry: WhatsAppMultiTenantRegistry, interval_s: float = 30.0
) -> None:
    """Background task: invoke registry.tick() every interval_s seconds.

    Survives transient errors (logs and continues). Exits cleanly on cancel.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            await registry.tick()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("wa_poll_loop iteration error: {}", e)

"""Agent tool: transcribe a local audio file via OpenAI.

Wraps `OpenAITranscriptionProvider` with two extras the agent loop needs:
- ffmpeg pre-compression to fit OpenAI's 25MB upload cap (mono 16k MP3 64k)
- transient `_progress` outbound events (start / 30s heartbeat / done) so
  long-running transcription doesn't go silent for several minutes.

Default model is `gpt-4o-transcribe-diarize` — sales-coach use case wants
speaker labels. Callers can override via the `model` parameter.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema
from nanobot.bus.events import OutboundMessage
from nanobot.providers.transcription import OpenAITranscriptionProvider

_OPENAI_AUDIO_SIZE_LIMIT_MB = 25
_COMPRESS_THRESHOLD_MB = 20
_DEFAULT_MODEL = "gpt-4o-transcribe-diarize"
_TRANSCRIBE_TIMEOUT_S = 600.0
_HEARTBEAT_INTERVAL_S = 30
_FORMATS_NEEDING_COMPRESS = frozenset({".wav", ".flac", ".aif", ".aiff", ".aac"})


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema(
            "Absolute path to a local audio file (WAV/MP3/M4A/etc.). Files "
            "over 20MB or in uncompressed formats are auto-transcoded to "
            "mono 16kHz MP3 to fit OpenAI's 25MB upload cap."
        ),
        model=StringSchema(
            "Optional override for the OpenAI transcription model. Default "
            "'gpt-4o-transcribe-diarize' returns speaker-labeled segments; "
            "pass 'gpt-4o-transcribe' for plain text (single-speaker use)."
        ),
        language=StringSchema(
            "Optional ISO-639 language hint (e.g. 'yue' for Cantonese, 'cmn' "
            "for Mandarin). Improves accuracy on non-English audio."
        ),
        required=["path"],
    )
)
class TranscribeAudioTool(Tool):
    """Transcribe a local audio file with speaker diarization.

    Output format: lines of `<speaker> mm:ss  <text>` (e.g. `A 00:03  Hello`).
    Single-speaker audio still gets an `A` label due to diarize model design —
    downstream prompts should infer speaker roles from context, not label.
    A sidecar `.txt` is written next to the source for debugging and re-use.
    """

    name = "transcribe_audio"
    description = (
        "Transcribe a local audio file with speaker diarization and timestamps. "
        "Use when the user provides a sales call recording, voice memo, podcast, "
        "or other audio that needs to be converted to text before analysis."
    )

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel: ContextVar[str] = ContextVar(
            "transcribe_audio_default_channel", default=default_channel
        )
        self._default_chat_id: ContextVar[str] = ContextVar(
            "transcribe_audio_default_chat_id", default=default_chat_id
        )

    def set_context(self, channel: str, chat_id: str) -> None:
        """Bind the channel/chat for any `_progress` events emitted by this turn."""
        self._default_channel.set(channel)
        self._default_chat_id.set(chat_id)

    async def _emit_progress(self, content: str) -> None:
        """No-op if no send_callback or context isn't bound."""
        if not self._send_callback:
            return
        channel = self._default_channel.get()
        chat_id = self._default_chat_id.get()
        if not channel or not chat_id:
            return
        try:
            await self._send_callback(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                media=[],
                buttons=[],
                metadata={"_progress": True},
            ))
        except Exception as e:
            logger.warning("transcribe_audio: progress emit failed: {}", e)

    async def _heartbeat(self) -> None:
        elapsed = 0
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            elapsed += _HEARTBEAT_INTERVAL_S
            await self._emit_progress(f"Still transcribing... ({elapsed}s elapsed)")

    async def _compress(self, src: Path, size_mb: float) -> tuple[Path | None, str | None]:
        if not shutil.which("ffmpeg"):
            return None, (
                f"Error: file is {size_mb:.1f}MB and requires compression to "
                f"fit OpenAI's {_OPENAI_AUDIO_SIZE_LIMIT_MB}MB cap, but ffmpeg "
                f"is not installed on the host."
            )
        await self._emit_progress(f"Compressing audio ({size_mb:.1f}MB)...")
        target = Path(tempfile.gettempdir()) / f"nanobot-stt-{uuid.uuid4().hex}.mp3"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", "16000", "-b:a", "64k",
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_tail = stderr.decode("utf-8", errors="replace")[-500:]
            return None, f"Error: ffmpeg compression failed: {err_tail}"
        return target, None

    async def execute(
        self,
        path: str,
        model: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> str:
        src = Path(path).expanduser()
        if not src.exists():
            return f"Error: audio file not found: {path}"
        if not src.is_file():
            return f"Error: not a regular file: {path}"

        size_mb = src.stat().st_size / 1024 / 1024
        needs_compress = (
            size_mb > _COMPRESS_THRESHOLD_MB
            or src.suffix.lower() in _FORMATS_NEEDING_COMPRESS
        )
        compressed_tmp: Path | None = None

        if needs_compress:
            compressed_tmp, err = await self._compress(src, size_mb)
            if err:
                return err
            audio_for_api = compressed_tmp
        else:
            audio_for_api = src

        await self._emit_progress(
            "Transcribing audio (typically 1-2 min per minute of recording)..."
        )
        provider = OpenAITranscriptionProvider(
            model=model or _DEFAULT_MODEL,
            language=language,
            timeout=_TRANSCRIBE_TIMEOUT_S,
        )

        heartbeat: asyncio.Task | None = None
        if self._send_callback and self._default_channel.get() and self._default_chat_id.get():
            heartbeat = asyncio.create_task(self._heartbeat())

        try:
            transcript = await provider.transcribe(audio_for_api)
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
            if compressed_tmp is not None and compressed_tmp.exists():
                try:
                    compressed_tmp.unlink()
                except OSError:
                    pass

        if not transcript:
            return (
                "Error: transcription returned empty output. "
                "Check OPENAI_API_KEY, network connectivity, and audio file integrity."
            )

        sidecar = src.with_suffix(src.suffix + ".txt")
        try:
            sidecar.write_text(transcript, encoding="utf-8")
        except OSError as e:
            logger.warning("transcribe_audio: sidecar write failed: {}", e)

        await self._emit_progress("Transcription complete. Analyzing...")
        return transcript

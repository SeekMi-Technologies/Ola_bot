"""Voice transcription providers (Groq and OpenAI Whisper)."""

import os
from pathlib import Path

import httpx
from loguru import logger


class OpenAITranscriptionProvider:
    """Voice transcription provider using OpenAI's audio API.

    Supports the gpt-4o-transcribe family (incl. -diarize variant which
    returns speaker-labeled timestamped segments via response_format=diarized_json)
    and the legacy whisper-1 model.

    Default is gpt-4o-transcribe (plain text output) — appropriate for
    single-speaker delivery channels (WhatsApp voice messages, Telegram, etc.).
    Callers needing speaker diarization (e.g. sales-coach pipeline analyzing
    multi-party calls) explicitly pass model="gpt-4o-transcribe-diarize".
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
        chunking_strategy: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = (
            api_base
            or os.environ.get("OPENAI_TRANSCRIPTION_BASE_URL")
            or "https://api.openai.com/v1/audio/transcriptions"
        )
        self.language = language or None
        self.model = (
            model
            or os.environ.get("OPENAI_TRANSCRIPTION_MODEL")
            or "gpt-4o-transcribe"
        )
        if "diarize" in self.model and chunking_strategy is None:
            chunking_strategy = "auto"
        self.chunking_strategy = chunking_strategy
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("OPENAI_TRANSCRIPTION_TIMEOUT") or "300")
        )

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenAI API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        is_diarize = "diarize" in self.model
        response_format = "diarized_json" if is_diarize else "json"

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files: dict = {
                        "file": (path.name, f),
                        "model": (None, self.model),
                        "response_format": (None, response_format),
                    }
                    if self.language:
                        files["language"] = (None, self.language)
                    if self.chunking_strategy:
                        files["chunking_strategy"] = (None, self.chunking_strategy)
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = await client.post(
                        self.api_url, headers=headers, files=files, timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if is_diarize:
                        return self._format_diarized(data)
                    return data.get("text", "")
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else "<no body>"
            logger.error(
                "OpenAI transcription HTTP {} ({}): {}",
                e.response.status_code if e.response is not None else "?",
                type(e).__name__,
                body[:500],
            )
            return ""
        except Exception as e:
            logger.error("OpenAI transcription error ({}): {!r}", type(e).__name__, e)
            return ""

    @staticmethod
    def _format_diarized(data: dict) -> str:
        """Render diarized_json segments as 'A 00:03  text' lines.

        Falls back to the raw `text` field if segments are missing or empty.
        """
        segments = data.get("segments") or []
        if not segments:
            return data.get("text", "")
        lines: list[str] = []
        for seg in segments:
            speaker = seg.get("speaker") or "?"
            start = float(seg.get("start") or 0.0)
            text = (seg.get("text") or "").strip()
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"{speaker} {mm:02d}:{ss:02d}  {text}")
        return "\n".join(lines)


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = api_base or os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1/audio/transcriptions"
        self.language = language or None

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    if self.language:
                        files["language"] = (None, self.language)
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""

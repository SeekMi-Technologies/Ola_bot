"""Voice transcription providers (Groq and OpenAI Whisper)."""

import os
from pathlib import Path

import httpx
from loguru import logger


class OpenAITranscriptionProvider:
    """Voice transcription provider using OpenAI's audio API.

    Designed for nanobot's real-time channel STT path (BaseChannel.transcribe_audio):
    single-speaker delivery channel voice messages — WhatsApp PTT, Telegram voice
    notes, WeChat voice, etc. Defaults to gpt-4o-transcribe (plain text). Bulk
    multi-speaker transcription (sales-coach calls, podcasts) is handled by Ola
    CRM's transcriptionWorker, not here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = (
            api_base
            or os.environ.get("OPENAI_TRANSCRIPTION_BASE_URL")
            or "https://api.openai.com/v1/audio/transcriptions"
        )
        self.language = language
        self.model = (
            model
            or os.environ.get("OPENAI_TRANSCRIPTION_MODEL")
            or "gpt-4o-transcribe"
        )
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

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files: dict = {
                        "file": (path.name, f),
                        "model": (None, self.model),
                        "response_format": (None, "json"),
                    }
                    if self.language:
                        files["language"] = (None, self.language)
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = await client.post(
                        self.api_url, headers=headers, files=files, timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
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

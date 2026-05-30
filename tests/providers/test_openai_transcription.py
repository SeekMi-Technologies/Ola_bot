"""Tests for OpenAITranscriptionProvider — real-time channel STT path.

Diarized output (multi-speaker labeled segments) was removed in May 2026;
bulk multi-speaker transcription now lives in Ola CRM's transcriptionWorker.
nanobot's provider only handles single-speaker delivery-channel audio.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nanobot.providers.transcription import OpenAITranscriptionProvider


def _make_mock_client(json_response: dict):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=json_response)

    client_instance = MagicMock()
    client_instance.post = AsyncMock(return_value=response)

    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client_instance)
    async_client.__aexit__ = AsyncMock(return_value=None)
    return async_client, client_instance


def test_init_default_model_is_plain_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.model == "gpt-4o-transcribe"


def test_init_explicit_whisper1() -> None:
    p = OpenAITranscriptionProvider(api_key="sk-test", model="whisper-1")
    assert p.model == "whisper-1"


def test_init_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.model == "gpt-4o-mini-transcribe"


def test_init_timeout_default_300s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_TIMEOUT", raising=False)
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.timeout == 300.0


def test_init_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_TIMEOUT", "60")
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.timeout == 60.0


def test_init_timeout_explicit_wins() -> None:
    p = OpenAITranscriptionProvider(api_key="sk-test", timeout=42.5)
    assert p.timeout == 42.5


async def test_transcribe_no_api_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAITranscriptionProvider(api_key=None)
    result = await p.transcribe("/nonexistent/file.wav")
    assert result == ""


async def test_transcribe_file_not_found_returns_empty(tmp_path: Path) -> None:
    p = OpenAITranscriptionProvider(api_key="sk-test")
    result = await p.transcribe(tmp_path / "nope.wav")
    assert result == ""


async def test_transcribe_default_plain_payload(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake audio")

    json_response = {"text": "Hello there, just confirming tomorrow."}
    async_client, client_instance = _make_mock_client(json_response)

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test")  # default model
        result = await p.transcribe(audio)

    files = client_instance.post.call_args.kwargs["files"]
    assert files["model"] == (None, "gpt-4o-transcribe")
    assert files["response_format"] == (None, "json")
    assert "chunking_strategy" not in files
    assert result == "Hello there, just confirming tomorrow."


async def test_transcribe_whisper1_legacy_path(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake audio")

    json_response = {"text": "plain transcript output"}
    async_client, client_instance = _make_mock_client(json_response)

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test", model="whisper-1")
        result = await p.transcribe(audio)

    files = client_instance.post.call_args.kwargs["files"]
    assert files["model"] == (None, "whisper-1")
    assert files["response_format"] == (None, "json")
    assert result == "plain transcript output"


async def test_transcribe_language_param_passed(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    async_client, client_instance = _make_mock_client({"text": ""})

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test", language="yue", model="whisper-1")
        await p.transcribe(audio)

    files = client_instance.post.call_args.kwargs["files"]
    assert files["language"] == (None, "yue")


async def test_transcribe_api_error_returns_empty(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=Exception("HTTP 500"))
    client_instance = MagicMock()
    client_instance.post = AsyncMock(return_value=response)
    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client_instance)
    async_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test")
        result = await p.transcribe(audio)

    assert result == ""


async def test_transcribe_http_status_error_returns_empty(tmp_path: Path) -> None:
    """HTTPStatusError has a dedicated except branch that logs response body
    + status — covered separately from the generic Exception path."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    err_response = MagicMock()
    err_response.status_code = 429
    err_response.text = "rate limited"
    response = MagicMock()
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("rate limited", request=MagicMock(), response=err_response)
    )
    client_instance = MagicMock()
    client_instance.post = AsyncMock(return_value=response)
    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client_instance)
    async_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test")
        result = await p.transcribe(audio)

    assert result == ""

"""Tests for OpenAITranscriptionProvider with gpt-4o-transcribe-diarize support."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_format_diarized_renders_segments() -> None:
    data = {
        "text": "ignored when segments present",
        "segments": [
            {"speaker": "A", "start": 3.412, "end": 5.312, "text": "Hello there"},
            {"speaker": "B", "start": 65.0, "end": 67.0, "text": "Hi back"},
        ],
    }
    result = OpenAITranscriptionProvider._format_diarized(data)
    assert result.split("\n") == ["A 00:03  Hello there", "B 01:05  Hi back"]


def test_format_diarized_empty_segments_falls_back_to_text() -> None:
    assert OpenAITranscriptionProvider._format_diarized(
        {"text": "fallback", "segments": []}
    ) == "fallback"


def test_format_diarized_missing_segments_falls_back_to_text() -> None:
    assert OpenAITranscriptionProvider._format_diarized({"text": "fallback"}) == "fallback"


def test_format_diarized_defensive_against_missing_fields() -> None:
    data = {"segments": [{}, {"speaker": "A", "text": "ok"}]}
    result = OpenAITranscriptionProvider._format_diarized(data)
    assert result == "? 00:00  \nA 00:00  ok"


def test_init_default_model_is_diarize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.model == "gpt-4o-transcribe-diarize"
    assert p.chunking_strategy == "auto"


def test_init_explicit_whisper1_no_chunking() -> None:
    p = OpenAITranscriptionProvider(api_key="sk-test", model="whisper-1")
    assert p.model == "whisper-1"
    assert p.chunking_strategy is None


def test_init_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    p = OpenAITranscriptionProvider(api_key="sk-test")
    assert p.model == "gpt-4o-mini-transcribe"
    assert p.chunking_strategy is None


def test_init_explicit_chunking_strategy_respected() -> None:
    p = OpenAITranscriptionProvider(
        api_key="sk-test", model="gpt-4o-transcribe-diarize", chunking_strategy="server_vad"
    )
    assert p.chunking_strategy == "server_vad"


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


async def test_transcribe_diarize_sends_correct_payload(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake audio bytes")

    json_response = {
        "text": "Hello",
        "segments": [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"}],
    }
    async_client, client_instance = _make_mock_client(json_response)

    with patch(
        "nanobot.providers.transcription.httpx.AsyncClient", return_value=async_client
    ):
        p = OpenAITranscriptionProvider(api_key="sk-test")
        result = await p.transcribe(audio)

    assert client_instance.post.called
    call_kwargs = client_instance.post.call_args.kwargs
    files = call_kwargs["files"]
    assert files["model"] == (None, "gpt-4o-transcribe-diarize")
    assert files["response_format"] == (None, "diarized_json")
    assert files["chunking_strategy"] == (None, "auto")
    assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert result == "A 00:00  Hello"


async def test_transcribe_whisper1_legacy_path_no_chunking(tmp_path: Path) -> None:
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
    assert "chunking_strategy" not in files
    assert result == "plain transcript output"


async def test_transcribe_language_param_passed(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    async_client, client_instance = _make_mock_client({"text": "", "segments": []})

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

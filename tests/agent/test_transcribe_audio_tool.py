"""Tests for nanobot.agent.tools.audio.TranscribeAudioTool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.audio import TranscribeAudioTool
from nanobot.bus.events import OutboundMessage


def _make_provider_mock(transcript: str = "A 00:00  hello") -> MagicMock:
    provider = MagicMock()
    provider.transcribe = AsyncMock(return_value=transcript)
    return provider


def _make_subprocess_mock(returncode: int = 0, stderr: bytes = b"") -> AsyncMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return AsyncMock(return_value=proc)


def test_tool_schema_basics() -> None:
    tool = TranscribeAudioTool()
    assert tool.name == "transcribe_audio"
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["required"] == ["path"]


async def test_execute_file_not_found(tmp_path: Path) -> None:
    tool = TranscribeAudioTool()
    result = await tool.execute(path=str(tmp_path / "missing.mp3"))
    assert result.startswith("Error: audio file not found")


async def test_execute_path_is_directory(tmp_path: Path) -> None:
    tool = TranscribeAudioTool()
    result = await tool.execute(path=str(tmp_path))
    assert result.startswith("Error: not a regular file")


async def test_execute_small_mp3_skips_compression(tmp_path: Path) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)  # tiny mp3-shaped bytes

    provider = _make_provider_mock("A 00:00  hello world")

    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(audio))

    assert result == "A 00:00  hello world"
    # Provider was called with the original file, no compression intermediate
    call_args = provider.transcribe.call_args
    assert call_args.args[0] == audio
    # Sidecar written next to source
    sidecar = audio.with_suffix(".mp3.txt")
    assert sidecar.exists()
    assert sidecar.read_text() == "A 00:00  hello world"


async def test_execute_wav_triggers_compression(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    provider = _make_provider_mock("A 00:00  hi")
    subprocess_factory = _make_subprocess_mock(returncode=0)

    def _fake_compressed(dst_path: str) -> None:
        Path(dst_path).write_bytes(b"compressed-mp3-bytes")

    async def _fake_communicate() -> tuple[bytes, bytes]:
        # The mock subprocess "creates" the output file the tool expects.
        # We grab the target path from the create_subprocess_exec call args.
        return (b"", b"")

    with patch("nanobot.agent.tools.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch(
             "nanobot.agent.tools.audio.asyncio.create_subprocess_exec",
             subprocess_factory,
         ), \
         patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool()
        # Stub: ffmpeg subprocess should create the tmp file. Since we mocked it,
        # we need to write the file ourselves matching the tmp path the tool generated.
        # Simpler: patch Path.exists for the tmp to return True.
        with patch.object(Path, "unlink", lambda self, *a, **kw: None):
            result = await tool.execute(path=str(audio))

    assert result == "A 00:00  hi"
    assert subprocess_factory.called
    cmd_args = subprocess_factory.call_args.args
    assert cmd_args[0] == "ffmpeg"
    assert "-ac" in cmd_args and "1" in cmd_args
    assert "-ar" in cmd_args and "16000" in cmd_args


async def test_execute_no_ffmpeg_for_wav(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    with patch("nanobot.agent.tools.audio.shutil.which", return_value=None):
        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(audio))

    assert "ffmpeg" in result
    assert "not installed" in result


async def test_execute_ffmpeg_subprocess_failure(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    subprocess_factory = _make_subprocess_mock(
        returncode=1, stderr=b"ffmpeg: invalid data found"
    )

    with patch("nanobot.agent.tools.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch(
             "nanobot.agent.tools.audio.asyncio.create_subprocess_exec",
             subprocess_factory,
         ):
        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(audio))

    assert result.startswith("Error: ffmpeg compression failed")
    assert "invalid data found" in result


async def test_execute_empty_provider_response(tmp_path: Path) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)
    provider = _make_provider_mock(transcript="")

    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool()
        result = await tool.execute(path=str(audio))

    assert result.startswith("Error: transcription returned empty output")
    sidecar = audio.with_suffix(".mp3.txt")
    assert not sidecar.exists()


async def test_execute_model_override_passes_to_provider(tmp_path: Path) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)
    provider = _make_provider_mock("plain text out")

    factory = MagicMock(return_value=provider)
    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", factory):
        tool = TranscribeAudioTool()
        await tool.execute(
            path=str(audio), model="gpt-4o-transcribe", language="yue"
        )

    factory.assert_called_once()
    kw = factory.call_args.kwargs
    assert kw["model"] == "gpt-4o-transcribe"
    assert kw["language"] == "yue"


async def test_execute_default_model_is_diarize(tmp_path: Path) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)
    provider = _make_provider_mock("A 00:00  hi")

    factory = MagicMock(return_value=provider)
    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", factory):
        tool = TranscribeAudioTool()
        await tool.execute(path=str(audio))

    assert factory.call_args.kwargs["model"] == "gpt-4o-transcribe-diarize"


async def test_emit_progress_no_callback_is_noop() -> None:
    tool = TranscribeAudioTool()  # no callback
    await tool._emit_progress("anything")  # must not raise


async def test_emit_progress_no_context_is_noop() -> None:
    callback = AsyncMock()
    tool = TranscribeAudioTool(send_callback=callback)
    # No channel/chat_id set
    await tool._emit_progress("anything")
    callback.assert_not_called()


async def test_emit_progress_happy_path() -> None:
    callback = AsyncMock()
    tool = TranscribeAudioTool(send_callback=callback)
    tool.set_context(channel="askola", chat_id="user-42")
    await tool._emit_progress("transcribing...")

    callback.assert_awaited_once()
    msg: OutboundMessage = callback.call_args.args[0]
    assert msg.channel == "askola"
    assert msg.chat_id == "user-42"
    assert msg.content == "transcribing..."
    assert msg.metadata.get("_progress") is True


async def test_execute_emits_progress_when_context_bound(tmp_path: Path) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)
    provider = _make_provider_mock("A 00:00  hi")
    callback = AsyncMock()

    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool(send_callback=callback)
        tool.set_context(channel="askola", chat_id="u1")
        await tool.execute(path=str(audio))

    # At minimum: "Transcribing audio..." (start) and "Transcription complete..." (end)
    progress_contents = [c.args[0].content for c in callback.await_args_list]
    assert any("Transcribing audio" in c for c in progress_contents)
    assert any("complete" in c.lower() for c in progress_contents)


async def test_heartbeat_fires_during_long_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)

    # Speed up heartbeat: 50ms instead of 30s
    monkeypatch.setattr("nanobot.agent.tools.audio._HEARTBEAT_INTERVAL_S", 0.05)

    async def _slow_transcribe(*args, **kwargs) -> str:
        await asyncio.sleep(0.2)  # 4 heartbeat intervals
        return "A 00:00  done"

    provider = MagicMock()
    provider.transcribe = AsyncMock(side_effect=_slow_transcribe)
    callback = AsyncMock()

    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool(send_callback=callback)
        tool.set_context(channel="askola", chat_id="u1")
        result = await tool.execute(path=str(audio))

    assert result == "A 00:00  done"
    progress_contents = [c.args[0].content for c in callback.await_args_list]
    # Expect at least one heartbeat between start and done
    heartbeats = [c for c in progress_contents if "Still transcribing" in c]
    assert len(heartbeats) >= 1, f"no heartbeats fired; saw: {progress_contents}"


async def test_heartbeat_skipped_without_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If no send_callback or no context, heartbeat task isn't created."""
    audio = tmp_path / "small.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)
    monkeypatch.setattr("nanobot.agent.tools.audio._HEARTBEAT_INTERVAL_S", 0.05)

    async def _slow(*args, **kwargs) -> str:
        await asyncio.sleep(0.15)
        return "done"

    provider = MagicMock()
    provider.transcribe = AsyncMock(side_effect=_slow)

    with patch("nanobot.agent.tools.audio.OpenAITranscriptionProvider", return_value=provider):
        tool = TranscribeAudioTool()  # no callback
        result = await tool.execute(path=str(audio))

    assert result == "done"

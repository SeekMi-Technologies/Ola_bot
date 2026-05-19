"""Live integration tests for OpenAITranscriptionProvider.

Gated by OPENAI_API_KEY env var. Skipped when absent (so unit-only CI doesn't break).

Exercises the real OpenAI audio API end-to-end with a committed English fixture
(`tests/fixtures/sample-en.wav`, ~5s of macOS `say` output).

Cantonese quality is verified manually via `scripts/spike_cantonese_stt.py`
against the customer's actual recording (not committed).
"""

import os
import re
from pathlib import Path

import pytest

from nanobot.providers.transcription import OpenAITranscriptionProvider

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample-en.wav"

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live OpenAI transcription tests",
)


async def test_diarize_returns_speaker_timestamp_format() -> None:
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    provider = OpenAITranscriptionProvider(model="gpt-4o-transcribe-diarize")
    result = await provider.transcribe(FIXTURE)

    assert result, "empty result from live API"
    # diarized output must contain at least one "<speaker> mm:ss  text" line
    first_line = result.split("\n")[0]
    assert re.match(r"^\S+\s+\d{2}:\d{2}\s{2}.+", first_line), (
        f"first line does not match diarized format: {first_line!r}\n"
        f"full result:\n{result}"
    )


async def test_whisper1_legacy_returns_plain_text() -> None:
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    provider = OpenAITranscriptionProvider(model="whisper-1")
    result = await provider.transcribe(FIXTURE)

    assert result, "empty result from whisper-1"
    assert "\n" not in result.strip() or "00:" not in result, (
        "whisper-1 path should return plain text, not diarized format"
    )

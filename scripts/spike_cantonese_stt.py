#!/usr/bin/env python3
"""Manual ear-test spike for Cantonese STT quality (Phase 0 of issue #249).

Runs the patched OpenAITranscriptionProvider against a local audio file
(e.g. Gingersoft's customer recording) and dumps the diarized transcript
to stdout + a sidecar .txt file next to the input.

Usage:
    python scripts/spike_cantonese_stt.py <audio_path> [--model MODEL] [--lang yue]

Requires OPENAI_API_KEY in env (source .secrets/SERVERS.env in the CRM repo
to pick it up).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanobot.providers.transcription import OpenAITranscriptionProvider


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="path to audio file (WAV/MP3/M4A)")
    parser.add_argument(
        "--model",
        default="gpt-4o-transcribe-diarize",
        help="OpenAI transcription model (default: gpt-4o-transcribe-diarize)",
    )
    parser.add_argument(
        "--lang",
        default="yue",
        help="ISO-639 language hint (default: yue for Cantonese)",
    )
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"ERROR: file not found: {args.audio_path}", file=sys.stderr)
        return 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in env", file=sys.stderr)
        return 1

    size_mb = args.audio_path.stat().st_size / 1024 / 1024
    print(f"Audio:    {args.audio_path}")
    print(f"Size:     {size_mb:.1f} MB")
    print(f"Model:    {args.model}")
    print(f"Language: {args.lang}")
    print("Sending to OpenAI...")

    provider = OpenAITranscriptionProvider(
        model=args.model, language=args.lang, timeout=600.0
    )

    t0 = time.time()
    result = await provider.transcribe(args.audio_path)
    elapsed = time.time() - t0

    if not result:
        print(f"FAIL: empty result (elapsed {elapsed:.1f}s)", file=sys.stderr)
        return 2

    sidecar = args.audio_path.with_suffix(args.audio_path.suffix + ".txt")
    sidecar.write_text(result, encoding="utf-8")

    print(f"\n--- TRANSCRIPT (elapsed {elapsed:.1f}s, {len(result)} chars) ---\n")
    print(result)
    print(f"\n--- written to {sidecar} ---")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

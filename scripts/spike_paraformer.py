#!/usr/bin/env python3
"""Phase 0 spike for Paraformer-v2 long-form Cantonese/English STT (issue #257).

Tests Alibaba DashScope's paraformer-v2 model as a long-form alternative to
OpenAI gpt-4o-transcribe-diarize. Paraformer accepts up to 12h audio (2h with
diarization on) with native speaker_id per sentence, which — if quality is
comparable on Cantonese-English mix — eliminates the chunking + Jaccard
speaker-alignment plan in #257's original design.

The script intentionally takes a PUBLIC URL (https:// or oss://), not a local
path. File hosting is out of scope for Phase 0 — upload manually first:

    # via ossutil (recommended, keeps audio in our infra)
    ossutil cp ~/Desktop/Sophie26-5-15_副本.WAV oss://ola-spike/sophie.wav
    ossutil sign oss://ola-spike/sophie.wav --timeout 3600

Then run:

    export DASHSCOPE_API_KEY=sk-...
    python scripts/spike_paraformer.py "<signed-https-url>" \\
        --lang yue,zh,en --speakers 2 \\
        --out /tmp/sophie.paraformer.txt

Output sidecar shape matches OpenAITranscriptionProvider._format_diarized
(`SPEAKER_N mm:ss  text` lines) so it can be diff'd directly against an
OpenAI run on the same file (use spike_cantonese_stt.py for the OpenAI side).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
TASK_URL_TEMPLATE = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
POLL_INTERVAL_SEC = 5
POLL_MAX_WAIT_SEC = 1800


def format_segments(transcripts: list[dict]) -> str:
    """Render Paraformer sentences as 'SPEAKER_N mm:ss  text'.

    Mirrors OpenAITranscriptionProvider._format_diarized so spike output is
    diff-able against existing OpenAI sidecars line-for-line.
    """
    lines: list[str] = []
    for tr in transcripts:
        for sent in tr.get("sentences") or []:
            spk = sent.get("speaker_id")
            speaker = f"SPEAKER_{spk}" if spk is not None else "?"
            begin_ms = int(sent.get("begin_time") or 0)
            start = begin_ms / 1000.0
            text = (sent.get("text") or "").strip()
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"{speaker} {mm:02d}:{ss:02d}  {text}")
    return "\n".join(lines)


async def submit_task(
    client: httpx.AsyncClient,
    api_key: str,
    file_url: str,
    lang_hints: list[str],
    speaker_count: int,
) -> str:
    body = {
        "model": "paraformer-v2",
        "input": {"file_urls": [file_url]},
        "parameters": {
            "channel_id": [0],
            "language_hints": lang_hints,
            "diarization_enabled": True,
            "speaker_count": speaker_count,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    resp = await client.post(SUBMIT_URL, json=body, headers=headers, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"submit returned no task_id: {data}")
    return task_id


async def poll_task(client: httpx.AsyncClient, api_key: str, task_id: str) -> dict:
    url = TASK_URL_TEMPLATE.format(task_id=task_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + POLL_MAX_WAIT_SEC
    last_status = None
    while time.time() < deadline:
        resp = await client.post(url, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        status = (data.get("output") or {}).get("task_status")
        if status != last_status:
            elapsed = int(POLL_MAX_WAIT_SEC - (deadline - time.time()))
            print(f"  [{elapsed:>4}s] task_status={status}")
            last_status = status
        if status in ("SUCCEEDED", "FAILED"):
            return data
        await asyncio.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"task {task_id} did not finish within {POLL_MAX_WAIT_SEC}s")


async def fetch_transcripts(client: httpx.AsyncClient, transcription_url: str) -> list[dict]:
    resp = await client.get(transcription_url, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("transcripts") or []


async def run(file_url: str, lang_hints: list[str], speaker_count: int, out_path: Path) -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set in env", file=sys.stderr)
        return 1

    print(f"file_url:       {file_url}")
    print(f"lang_hints:     {lang_hints}")
    print(f"speaker_count:  {speaker_count}")
    print(f"out_path:       {out_path}")
    print()

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        print("Submitting Paraformer task...")
        task_id = await submit_task(client, api_key, file_url, lang_hints, speaker_count)
        print(f"  task_id={task_id}\n")

        print(f"Polling every {POLL_INTERVAL_SEC}s (max {POLL_MAX_WAIT_SEC}s)...")
        task = await poll_task(client, api_key, task_id)
        status = (task.get("output") or {}).get("task_status")
        if status != "SUCCEEDED":
            print(f"\nFAIL: task_status={status}", file=sys.stderr)
            print(task, file=sys.stderr)
            return 2

        results = (task.get("output") or {}).get("results") or []
        if not results:
            print(f"\nFAIL: SUCCEEDED but no results: {task}", file=sys.stderr)
            return 3
        sub = results[0]
        if sub.get("subtask_status") != "SUCCEEDED":
            print(f"\nFAIL: subtask_status={sub.get('subtask_status')}", file=sys.stderr)
            print(sub, file=sys.stderr)
            return 4

        transcription_url = sub.get("transcription_url")
        print(f"\nFetching transcripts from result OSS...")
        transcripts = await fetch_transcripts(client, transcription_url)

    elapsed = time.time() - t0
    rendered = format_segments(transcripts)
    if not rendered:
        print(f"\nFAIL: empty transcript (elapsed {elapsed:.1f}s)", file=sys.stderr)
        return 5

    out_path.write_text(rendered, encoding="utf-8")
    sentence_count = sum(len(t.get("sentences") or []) for t in transcripts)
    speaker_set = sorted(
        {
            s.get("speaker_id")
            for t in transcripts
            for s in (t.get("sentences") or [])
            if s.get("speaker_id") is not None
        }
    )

    print()
    print(f"--- DONE in {elapsed:.1f}s ---")
    print(f"sentences:        {sentence_count}")
    print(f"speakers seen:    {speaker_set}")
    print(f"sidecar bytes:    {len(rendered)}")
    print(f"written to:       {out_path}")
    print()
    print("--- first 20 lines ---")
    for line in rendered.splitlines()[:20]:
        print(line)
    if sentence_count > 20:
        print(f"... ({sentence_count - 20} more lines in sidecar)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file_url",
        help="public HTTPS or oss:// URL to the audio file (see docstring for upload steps)",
    )
    parser.add_argument(
        "--lang",
        default="yue,zh,en",
        help="comma-separated language hints, default yue,zh,en for Cantonese+Mandarin+English",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="expected speaker count hint, default 2 (sales + customer)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output sidecar path (default: ./<basename>.paraformer.txt)",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        basename = Path(urlparse(args.file_url).path).stem or "spike"
        out_path = Path.cwd() / f"{basename}.paraformer.txt"

    lang_hints = [s.strip() for s in args.lang.split(",") if s.strip()]
    return asyncio.run(run(args.file_url, lang_hints, args.speakers, out_path))


if __name__ == "__main__":
    sys.exit(main())

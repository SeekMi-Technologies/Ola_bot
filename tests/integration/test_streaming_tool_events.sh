#!/usr/bin/env bash
# Integration tests for streaming tool_events in /v1/chat/completions
# (Ola CRM issue #131, backlog L1).
#
# Unlike the pytest suite in tests/test_api_stream.py (which mocks the
# agent), this script exercises the REAL stack: NanoBot + MCP + Mongo +
# Gemini. It catches integration bugs that mock-based tests can't, e.g.
# the L1 `on_stream_end(resuming=True)` issue that only manifests when
# the actual agent loop fires the resume-pause callback during a real
# tool iteration.
#
# Prerequisites:
#   - CRM stack running: bash ~/Documents/GitHub/crm/start-dev.sh
#   - NanoBot listening on 127.0.0.1:8900
#   - MCP server on 127.0.0.1:8889 with customer/merch/quote tools registered
#   - Gemini API key valid
#
# Usage:
#   bash tests/integration/test_streaming_tool_events.sh
#
# Exit code: 0 if all scenarios PASS, 1 if any FAIL.
#
# Scenarios:
#   A. Tool-driven response: prompt that triggers at least one MCP tool call →
#      stream contains both event:tool_event frames and text deltas in order.
#   B. (Skipped here — covered by unit test test_tool_event_with_error_phase_propagates.
#       Real-stack tool-error testing tracked as L1-TD.)
#   C. Pure text response: prompt that does NOT need tools → stream contains
#      only text deltas, no event:tool_event frames.
#   D. Non-stream regression: stream=false still returns metadata.tool_events
#       JSON envelope (the pre-L1 behavior must not regress).

set -u

NANOBOT_URL="${NANOBOT_URL:-http://127.0.0.1:8900}"
TIMEOUT_SECS=90
TMPDIR_BASE="${TMPDIR:-/tmp}"
WORKDIR="$(mktemp -d "$TMPDIR_BASE/nanobot-integ-XXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

PASSES=0
FAILS=0

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

echo "==> Pre-flight: NanoBot reachable at $NANOBOT_URL?"
if ! curl -sf -o /dev/null --max-time 5 "$NANOBOT_URL/v1/models"; then
  red "FAIL: cannot reach $NANOBOT_URL/v1/models"
  echo "     Is NanoBot running? Try: bash ~/Documents/GitHub/crm/start-dev.sh"
  exit 1
fi
green "    OK"

# ---------------------------------------------------------------------------
# Scenario A — tool-driven response
# ---------------------------------------------------------------------------

echo
echo "==> Scenario A: tool-driven response triggers event:tool_event frames"

OUT_A="$WORKDIR/a.sse"
curl -sN -X POST "$NANOBOT_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --max-time "$TIMEOUT_SECS" \
  -d '{
    "messages": [{"role":"user","content":"Please use the merch.search tool to find products containing the word stainless"}],
    "stream": true,
    "session_id": "integ-a"
  }' > "$OUT_A" 2>&1

A_TOOL_FRAMES=$(grep -c '^event: tool_event' "$OUT_A" || true)
A_TEXT_DELTAS=$(grep -c 'delta": {"content"' "$OUT_A" || true)
A_HAS_DONE=$(grep -c '\[DONE\]' "$OUT_A" || true)
A_HAS_START=$(grep -c '"phase": "start"' "$OUT_A" || true)
A_HAS_END=$(grep -c '"phase": "end"' "$OUT_A" || true)

if [[ $A_TOOL_FRAMES -ge 2 && $A_TEXT_DELTAS -ge 1 && $A_HAS_DONE -ge 1 \
      && $A_HAS_START -ge 1 && $A_HAS_END -ge 1 ]]; then
  green "    PASS — tool_event frames=$A_TOOL_FRAMES, text deltas=$A_TEXT_DELTAS, [DONE]=yes, start/end phases present"
  PASSES=$((PASSES + 1))
else
  red "    FAIL — tool_event frames=$A_TOOL_FRAMES (>=2), text deltas=$A_TEXT_DELTAS (>=1), [DONE]=$A_HAS_DONE, start=$A_HAS_START, end=$A_HAS_END"
  echo "    See $OUT_A for full SSE output"
  FAILS=$((FAILS + 1))
fi

# ---------------------------------------------------------------------------
# Scenario C — pure text (no tool)
# ---------------------------------------------------------------------------

echo
echo "==> Scenario C: pure text response emits NO event:tool_event frames"

OUT_C="$WORKDIR/c.sse"
curl -sN -X POST "$NANOBOT_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --max-time "$TIMEOUT_SECS" \
  -d '{
    "messages": [{"role":"user","content":"Reply with the single word: pong. Do not call any tools."}],
    "stream": true,
    "session_id": "integ-c"
  }' > "$OUT_C" 2>&1

C_TOOL_FRAMES=$(grep -c '^event: tool_event' "$OUT_C" || true)
C_TEXT_DELTAS=$(grep -c 'delta": {"content"' "$OUT_C" || true)
C_HAS_DONE=$(grep -c '\[DONE\]' "$OUT_C" || true)

if [[ $C_TOOL_FRAMES -eq 0 && $C_TEXT_DELTAS -ge 1 && $C_HAS_DONE -ge 1 ]]; then
  green "    PASS — no tool_event frames (as expected), text deltas=$C_TEXT_DELTAS, [DONE]=yes"
  PASSES=$((PASSES + 1))
else
  red "    FAIL — tool_event frames=$C_TOOL_FRAMES (==0), text deltas=$C_TEXT_DELTAS (>=1), [DONE]=$C_HAS_DONE"
  echo "    See $OUT_C for full SSE output"
  FAILS=$((FAILS + 1))
fi

# ---------------------------------------------------------------------------
# Scenario D — non-stream regression
# ---------------------------------------------------------------------------

echo
echo "==> Scenario D: stream=false still returns metadata.tool_events JSON"

OUT_D="$WORKDIR/d.json"
curl -s -X POST "$NANOBOT_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --max-time "$TIMEOUT_SECS" \
  -d '{
    "messages": [{"role":"user","content":"Please use the merch.search tool to find products containing the word stainless"}],
    "stream": false,
    "session_id": "integ-d"
  }' > "$OUT_D" 2>&1

D_OBJECT=$(python3 -c "import json,sys; d=json.load(open('$OUT_D')); print(d.get('object',''))" 2>/dev/null || echo "")
D_TOOL_EVENTS_LEN=$(python3 -c "import json,sys; d=json.load(open('$OUT_D')); print(len(d.get('metadata',{}).get('tool_events',[])))" 2>/dev/null || echo "0")

if [[ "$D_OBJECT" == "chat.completion" && "$D_TOOL_EVENTS_LEN" -ge 2 ]]; then
  green "    PASS — object=chat.completion, metadata.tool_events len=$D_TOOL_EVENTS_LEN"
  PASSES=$((PASSES + 1))
else
  red "    FAIL — object='$D_OBJECT' (==chat.completion), tool_events len=$D_TOOL_EVENTS_LEN (>=2)"
  echo "    See $OUT_D for full response body"
  FAILS=$((FAILS + 1))
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo
echo "==================================================================="
if [[ $FAILS -eq 0 ]]; then
  green "ALL $PASSES SCENARIOS PASSED"
  exit 0
else
  red "FAILED: $FAILS scenario(s) failed, $PASSES passed"
  yellow "Artifacts kept at $WORKDIR until process exit"
  trap - EXIT
  exit 1
fi

"""Agent behavior tests — comprehensive multi-file analysis (issue #388).

Verifies that the runner correctly handles the tool-call pattern that
SOUL.md's "Comprehensive multi-file analysis" section relies on:

  file.search → [file.get_transcript(A), file.get_transcript(B), file.get_transcript(C)]
               (all three emitted in ONE LLM iteration → concurrent execution)
             → final synthesis text

Tests do NOT call a real LLM. Instead they drive AgentRunner directly with
a scripted provider that returns pre-planned tool calls, asserting that:

  1. All file.get_transcript calls that the LLM emits in one iteration are
     executed (concurrent_tools=True works end-to-end for this flow).
  2. The runner collects all results before the LLM makes its final text call.
  3. SOUL.md structurally contains the required comprehensive-analysis section.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.providers.base import LLMResponse, ToolCallRequest

# nanobot repo root is parents[2]; Ola workspace is a sibling of nanobot
# nanobot/tests/agent/test_comprehensive_analysis.py
#   parents[0] = nanobot/tests/agent/
#   parents[1] = nanobot/tests/
#   parents[2] = nanobot/
#   parents[3] = SeekMi_Tech/
SOUL_PATH = (
    Path(__file__).parents[3]
    / "Ola"
    / "ola"
    / "nanobot-workspace"
    / "SOUL.md"
)

_MAX_TOOL_RESULT_CHARS = 100_000


def _soul() -> str:
    if not SOUL_PATH.exists():
        pytest.skip(f"SOUL.md not found at {SOUL_PATH}")
    return SOUL_PATH.read_text(encoding="utf-8")


def _make_tools_mock(execute_fn) -> MagicMock:
    """Return a ToolRegistry-compatible mock with a custom execute coroutine."""
    m = MagicMock()
    m.get_definitions.return_value = []
    m.execute = execute_fn
    return m


def _make_provider(scripted_chat) -> MagicMock:
    provider = MagicMock()
    provider.chat_with_retry = scripted_chat
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096
    return provider


# ---------------------------------------------------------------------------
# SOUL.md structural contract
# ---------------------------------------------------------------------------


class TestSoulComprehensiveSection:
    """Pin the load-bearing anchors of the comprehensive-analysis prompt rule."""

    def test_section_exists(self):
        assert "## Comprehensive multi-file analysis" in _soul()

    def test_path_a_batch_instruction(self):
        assert "one single LLM iteration" in _soul()

    def test_path_b_no_file_tool_calls(self):
        soul = _soul()
        assert "Path B" in soul
        assert "No file.* tool calls" in soul

    def test_report_requires_three_sections(self):
        soul = _soul()
        for section in ("共同主题", "关键差异", "综合结论"):
            assert section in soul, f"missing required report section: {section}"

    def test_hard_rules_present(self):
        soul = _soul()
        for rule in (
            "No per-file progress commentary",
            "No duplicate tool calls",
            "No invented content",
            "No mixing paths",
        ):
            assert rule in soul, f"missing hard rule: {rule}"


# ---------------------------------------------------------------------------
# Runner batch-tool-call behavior
# ---------------------------------------------------------------------------


class TestRunnerBatchToolCalls:
    """Verify AgentRunner executes all tool calls emitted in one iteration.

    This is the mechanical precondition that SOUL.md's Path A relies on:
    when the LLM emits file.get_transcript for N files in one response,
    all N calls must be executed before the next LLM iteration.
    """

    @pytest.mark.asyncio
    async def test_all_get_transcript_calls_executed_in_one_iteration(self):
        """LLM emits 3 file.get_transcript calls in iteration 2 → all 3 execute."""
        file_ids = ["file-a", "file-b", "file-c"]
        executed: list[str] = []
        iteration = {"n": 0}

        async def scripted_chat(**_):
            iteration["n"] += 1
            if iteration["n"] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call-search",
                            name="file.search",
                            arguments={"status": "done"},
                        )
                    ],
                    usage={},
                )
            if iteration["n"] == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id=f"call-{fid}",
                            name="file.get_transcript",
                            arguments={"fileId": fid},
                        )
                        for fid in file_ids
                    ],
                    usage={},
                )
            return LLMResponse(
                content="综合结论：三段录音均涉及割嘴产品报价，客户关切价格与交期。",
                tool_calls=[],
                usage={},
            )

        async def _execute(name, arguments):
            if name == "file.search":
                return (
                    '{"ok":true,"data":{"found":true,"count":3,"files":['
                    '{"fileId":"file-a","originalName":"rec1.wav",'
                    '"transcription":{"status":"done"}},'
                    '{"fileId":"file-b","originalName":"rec2.wav",'
                    '"transcription":{"status":"done"}},'
                    '{"fileId":"file-c","originalName":"rec3.wav",'
                    '"transcription":{"status":"done"}}'
                    "]}}"
                )
            fid = arguments.get("fileId", "")
            executed.append(fid)
            return f'{{"ok":true,"data":{{"transcript":"content of {fid}"}}}}'

        runner = AgentRunner(_make_provider(scripted_chat))
        result = await runner.run(
            AgentRunSpec(
                initial_messages=[
                    {"role": "system", "content": "You are Ola."},
                    {"role": "user", "content": "帮我综合评估一下所有录音"},
                ],
                tools=_make_tools_mock(_execute),
                model="test-model",
                max_iterations=10,
                max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
                concurrent_tools=True,
            )
        )

        assert set(executed) == {"file-a", "file-b", "file-c"}, (
            f"Expected all 3 fileIds fetched, got: {executed}"
        )
        assert result.final_content is not None
        assert "综合结论" in result.final_content
        assert iteration["n"] == 3, f"Expected 3 LLM iterations, got {iteration['n']}"

    @pytest.mark.asyncio
    async def test_no_output_before_all_transcripts_collected(self):
        """Collection phase (iterations 1-2) must produce no text content."""
        iteration = {"n": 0}

        async def scripted_chat(**_):
            iteration["n"] += 1
            if iteration["n"] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call-search",
                            name="file.search",
                            arguments={"status": "done"},
                        )
                    ],
                    usage={},
                )
            if iteration["n"] == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call-ta",
                            name="file.get_transcript",
                            arguments={"fileId": "file-a"},
                        ),
                        ToolCallRequest(
                            id="call-tb",
                            name="file.get_transcript",
                            arguments={"fileId": "file-b"},
                        ),
                    ],
                    usage={},
                )
            return LLMResponse(
                content="综合结论：两段录音均涉及客户询价。",
                tool_calls=[],
                usage={},
            )

        async def _execute(name, arguments):
            if name == "file.search":
                return (
                    '{"ok":true,"data":{"found":true,"count":2,"files":['
                    '{"fileId":"file-a","originalName":"rec1.wav",'
                    '"transcription":{"status":"done"}},'
                    '{"fileId":"file-b","originalName":"rec2.wav",'
                    '"transcription":{"status":"done"}}'
                    "]}}"
                )
            fid = arguments.get("fileId", "")
            return f'{{"ok":true,"data":{{"transcript":"content of {fid}"}}}}'

        from nanobot.agent.hook import AgentHook, AgentHookContext

        captured_content: list[str] = []

        class CaptureHook(AgentHook):
            async def after_iteration(self, ctx: AgentHookContext) -> None:
                if ctx.response and ctx.response.content:
                    captured_content.append(ctx.response.content)

        runner = AgentRunner(_make_provider(scripted_chat))
        result = await runner.run(
            AgentRunSpec(
                initial_messages=[
                    {"role": "system", "content": "You are Ola."},
                    {"role": "user", "content": "综合评估所有录音"},
                ],
                tools=_make_tools_mock(_execute),
                model="test-model",
                max_iterations=10,
                max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
                concurrent_tools=True,
                hook=CaptureHook(),
            )
        )

        assert result.final_content == "综合结论：两段录音均涉及客户询价。"
        non_final = [c for c in captured_content if c != result.final_content]
        assert non_final == [], f"Unexpected intermediate text output: {non_final}"

    @pytest.mark.asyncio
    async def test_single_file_request_not_treated_as_comprehensive(self):
        """Single-file requests must not trigger the batch path."""
        executed_searches: list[str] = []
        iteration = {"n": 0}

        async def scripted_chat(**_):
            iteration["n"] += 1
            if iteration["n"] == 1:
                # Single get_transcript directly (no search first)
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call-single",
                            name="file.get_transcript",
                            arguments={"fileId": "file-x"},
                        )
                    ],
                    usage={},
                )
            return LLMResponse(
                content="这段录音讨论了 A-1473 的报价，客户需要 CIF Bangkok。",
                tool_calls=[],
                usage={},
            )

        async def _execute(name, *_):
            if name == "file.search":
                executed_searches.append(name)
            return '{"ok":true,"data":{"transcript":"content of file-x"}}'

        runner = AgentRunner(_make_provider(scripted_chat))
        result = await runner.run(
            AgentRunSpec(
                initial_messages=[
                    {"role": "system", "content": "You are Ola."},
                    {"role": "user", "content": "分析一下这段录音 file-x"},
                ],
                tools=_make_tools_mock(_execute),
                model="test-model",
                max_iterations=10,
                max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
                concurrent_tools=True,
            )
        )

        # file.search must NOT have been called (single-file path)
        assert executed_searches == [], (
            f"file.search should not be called for single-file requests, "
            f"but was called {len(executed_searches)} time(s)"
        )
        assert result.final_content is not None
        assert "综合结论" not in result.final_content

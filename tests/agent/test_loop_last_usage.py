"""Tests for AgentLoop._last_usage shape (Ola CRM #98 N2).

After each agent turn, `_last_usage` must be a flat 7-field dict that
nanobot/api/server.py can serialize directly into the `event: usage` SSE
frame (Ola CRM issue #98). Schema:

    {provider, model, prompt_tokens, completion_tokens, total_tokens,
     cached_tokens, iterations}
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop, _provider_name
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path, *, provider=None) -> AgentLoop:
    bus = MessageBus()
    if provider is None:
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


class TestProviderName:
    """_provider_name canonicalization for the usage frame."""

    def test_openai_compat_uses_spec_name(self):
        provider = MagicMock()
        provider._spec = MagicMock()
        provider._spec.name = "gemini"
        assert _provider_name(provider) == "gemini"

    def test_openai_compat_dashscope(self):
        provider = MagicMock()
        provider._spec = MagicMock()
        provider._spec.name = "dashscope"
        assert _provider_name(provider) == "dashscope"

    def test_anthropic_falls_to_class_name(self):
        class AnthropicProvider:
            pass
        provider = AnthropicProvider()
        assert _provider_name(provider) == "anthropic"

    def test_no_spec_unknown_class_falls_to_lowercased_class(self):
        class CustomFooProvider:
            pass
        provider = CustomFooProvider()
        assert _provider_name(provider) == "customfoo"

    def test_none_spec_falls_through(self):
        provider = MagicMock(spec=["_spec"])
        provider._spec = None
        # type(MagicMock()).__name__ == 'MagicMock' → 'magicmock' (no 'provider' suffix)
        assert _provider_name(provider) == "magicmock"

    def test_empty_spec_name_falls_through(self):
        provider = MagicMock()
        provider._spec = MagicMock()
        provider._spec.name = ""
        # falls through; magicmock (without spec=) class name path
        result = _provider_name(provider)
        assert result == "magicmock"


class TestLastUsageShape:
    """After _run_agent_loop, AgentLoop._last_usage has all 7 wire fields."""

    @pytest.mark.asyncio
    async def test_one_call_no_tools_populates_all_fields(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        loop.provider._spec = MagicMock()
        loop.provider._spec.name = "gemini"
        loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="hi back",
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        ))
        loop.tools.get_definitions = MagicMock(return_value=[])

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        u = loop._last_usage
        assert set(u.keys()) == {
            "provider", "model", "prompt_tokens", "completion_tokens",
            "total_tokens", "cached_tokens", "iterations",
        }
        assert u["provider"] == "gemini"
        assert u["model"] == "test-model"
        assert u["prompt_tokens"] == 100
        assert u["completion_tokens"] == 50
        assert u["total_tokens"] == 150
        assert u["cached_tokens"] == 0
        assert u["iterations"] == 1

    @pytest.mark.asyncio
    async def test_total_tokens_computed_when_provider_omits(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        loop.provider._spec = MagicMock()
        loop.provider._spec.name = "openai"
        loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="hello",
            tool_calls=[],
            usage={"prompt_tokens": 30, "completion_tokens": 12},  # no total_tokens
        ))
        loop.tools.get_definitions = MagicMock(return_value=[])

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert loop._last_usage["total_tokens"] == 42

    @pytest.mark.asyncio
    async def test_cached_tokens_surfaced(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        loop.provider._spec = MagicMock()
        loop.provider._spec.name = "anthropic"
        loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="hello",
            tool_calls=[],
            usage={
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
                "cached_tokens": 60,
            },
        ))
        loop.tools.get_definitions = MagicMock(return_value=[])

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert loop._last_usage["cached_tokens"] == 60

    @pytest.mark.asyncio
    async def test_tool_using_turn_iterations_two(self, tmp_path: Path):
        loop = _make_loop(tmp_path)
        loop.provider._spec = MagicMock()
        loop.provider._spec.name = "gemini"
        responses = iter([
            LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="c1", name="custom", arguments={})],
                usage={"prompt_tokens": 20, "completion_tokens": 5},
            ),
            LLMResponse(
                content="final",
                tool_calls=[],
                usage={"prompt_tokens": 25, "completion_tokens": 8},
            ),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        loop.tools.get_definitions = MagicMock(return_value=[])
        loop.tools.prepare_call = MagicMock(return_value=(None, {}, None))
        loop.tools.execute = AsyncMock(return_value="tool result")

        await loop._run_agent_loop([{"role": "user", "content": "do task"}])

        assert loop._last_usage["iterations"] == 2
        # Tokens accumulated across both iterations
        assert loop._last_usage["prompt_tokens"] == 45
        assert loop._last_usage["completion_tokens"] == 13

    @pytest.mark.asyncio
    async def test_anthropic_provider_class_name_fallback(self, tmp_path: Path):
        # Anthropic provider has no _spec attribute → _provider_name falls
        # through to the class-name branch and returns "anthropic". MagicMock
        # auto-attr mode would synthesize `_spec` on access (MagicMock!), so
        # we must explicitly suppress that with `spec=`. We then patch
        # __class__.__name__ to "AnthropicProvider" so _provider_name's class
        # check fires.
        provider = MagicMock()
        del provider._spec
        type(provider).__name__ = "AnthropicProvider"
        provider.get_default_model.return_value = "claude-test"
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="hi",
            tool_calls=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        ))

        loop = _make_loop(tmp_path, provider=provider)
        loop.tools.get_definitions = MagicMock(return_value=[])

        await loop._run_agent_loop([{"role": "user", "content": "hi"}])

        assert loop._last_usage["provider"] == "anthropic"

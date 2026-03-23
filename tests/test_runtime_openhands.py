"""Tests for OpenHands runtime implementation (mocked — no SDK dependency needed)."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, TextEvent
from fastharness.core.skill import Skill


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "name": "test",
        "description": "Test agent",
        "skills": [Skill(id="s1", name="S1", description="d")],
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


@pytest.fixture
def mock_openhands_sdk(monkeypatch):
    """Inject fake openhands.sdk modules so the runtime can import."""
    sdk_mod = ModuleType("openhands")
    sdk_sdk_mod = ModuleType("openhands.sdk")
    sdk_tool_mod = ModuleType("openhands.sdk.tool")

    mock_llm = MagicMock(name="LLM")
    mock_agent = MagicMock(name="Agent")
    mock_conversation_cls = MagicMock(name="Conversation")
    mock_tool = MagicMock(name="Tool")

    sdk_sdk_mod.LLM = mock_llm
    sdk_sdk_mod.Agent = mock_agent
    sdk_sdk_mod.Conversation = mock_conversation_cls
    sdk_tool_mod.Tool = mock_tool

    monkeypatch.setitem(sys.modules, "openhands", sdk_mod)
    monkeypatch.setitem(sys.modules, "openhands.sdk", sdk_sdk_mod)
    monkeypatch.setitem(sys.modules, "openhands.sdk.tool", sdk_tool_mod)

    # Clear any cached import of the runtime module
    monkeypatch.delitem(sys.modules, "fastharness.runtime.openhands", raising=False)

    return {
        "LLM": mock_llm,
        "Agent": mock_agent,
        "Conversation": mock_conversation_cls,
        "Tool": mock_tool,
    }


class TestOpenHandsRuntime:
    """Tests for OpenHandsRuntime run/stream behavior."""

    @pytest.mark.asyncio
    async def test_run_returns_text(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntime

        mock_conv = MagicMock()
        mock_conv.send_message = MagicMock()
        mock_conv.run = MagicMock()

        # Simulate conversation state with OpenHands event structure
        text_block = MagicMock()
        text_block.text = "Hello from OpenHands"
        llm_msg = MagicMock()
        llm_msg.content = [text_block]
        event = MagicMock()
        event.source = "agent"
        event.llm_message = llm_msg
        mock_state = MagicMock()
        mock_state.events = [event]
        mock_conv.state = mock_state

        runtime = OpenHandsRuntime(mock_conv)
        result = await runtime.run("Hi")

        assert result == "Hello from OpenHands"
        mock_conv.send_message.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_run_returns_empty_when_no_events(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntime

        mock_conv = MagicMock()
        mock_conv.send_message = MagicMock()
        mock_conv.run = MagicMock()
        mock_conv.state = MagicMock()
        mock_conv.state.events = []

        runtime = OpenHandsRuntime(mock_conv)
        result = await runtime.run("Hi")

        assert result == ""

    @pytest.mark.asyncio
    async def test_stream_yields_text_and_done(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntime

        mock_conv = MagicMock()
        mock_conv.send_message = MagicMock()
        mock_conv.run = MagicMock()

        text_block = MagicMock()
        text_block.text = "Streaming response"
        llm_msg = MagicMock()
        llm_msg.content = [text_block]
        event = MagicMock()
        event.source = "agent"
        event.llm_message = llm_msg
        mock_conv.state = MagicMock()
        mock_conv.state.events = [event]

        runtime = OpenHandsRuntime(mock_conv)
        events = []
        async for e in runtime.stream("test"):
            events.append(e)

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "Streaming response"
        assert isinstance(events[1], DoneEvent)
        assert events[1].final_text == "Streaming response"

    @pytest.mark.asyncio
    async def test_aclose_is_noop(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntime

        runtime = OpenHandsRuntime(MagicMock())
        await runtime.aclose()  # Should not raise


class TestOpenHandsRuntimeFactory:
    """Tests for OpenHandsRuntimeFactory lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_new_session(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntime, OpenHandsRuntimeFactory

        factory = OpenHandsRuntimeFactory(ttl_minutes=15)
        config = _make_config(tools=["terminal", "file_editor"])

        runtime = await factory.get_or_create("s1", config)

        assert isinstance(runtime, OpenHandsRuntime)
        mock_openhands_sdk["Agent"].assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_existing_session(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        factory = OpenHandsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s1", config)

        # Agent should only be created once
        assert mock_openhands_sdk["Agent"].call_count == 1

    @pytest.mark.asyncio
    async def test_remove_session(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        factory = OpenHandsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.remove("s1")

        assert "s1" not in factory._sessions

    @pytest.mark.asyncio
    async def test_shutdown_clears_all(self, mock_openhands_sdk):
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        factory = OpenHandsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s2", config)
        await factory.shutdown()

        assert len(factory._sessions) == 0

    @pytest.mark.asyncio
    async def test_stale_cleanup(self, mock_openhands_sdk):
        from datetime import timedelta

        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        factory = OpenHandsRuntimeFactory(ttl_minutes=1)
        config = _make_config()

        await factory.get_or_create("s1", config)

        # Backdate to make stale
        factory._sessions["s1"].last_accessed -= timedelta(minutes=2)

        async with factory._lock:
            stale = [k for k, v in factory._sessions.items() if v.is_stale(1)]
            for key in stale:
                factory._sessions.pop(key)

        assert "s1" not in factory._sessions

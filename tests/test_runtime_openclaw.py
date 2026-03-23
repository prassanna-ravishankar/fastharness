"""Tests for OpenClaw runtime implementation (mocked — no SDK dependency needed)."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, TextEvent, ToolEvent
from fastharness.core.skill import Skill


# Minimal stand-in classes that match OpenClaw SDK event type names
class ContentEvent:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolCallEvent:
    def __init__(self, tool: str, input: dict) -> None:
        self.tool = tool
        self.input = input


class DoneEventOC:
    """OpenClaw DoneEvent (distinct name to avoid clash with fastharness DoneEvent)."""

    def __init__(self, content=None, token_usage=None) -> None:
        self.content = content
        self.token_usage = token_usage

    # Make type(self).__name__ return "DoneEvent" for the runtime's dispatch
    pass


# Rename to match what the runtime checks
DoneEventOC.__name__ = "DoneEvent"


class ErrorEvent:
    def __init__(self, message: str) -> None:
        self.message = message


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "name": "test-agent",
        "description": "Test agent",
        "skills": [Skill(id="s1", name="S1", description="d")],
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


@pytest.fixture
def mock_openclaw_sdk(monkeypatch):
    """Inject fake openclaw_sdk module."""
    sdk_mod = ModuleType("openclaw_sdk")
    mock_client_cls = MagicMock(name="OpenClawClient")
    sdk_mod.OpenClawClient = mock_client_cls

    monkeypatch.setitem(sys.modules, "openclaw_sdk", sdk_mod)
    monkeypatch.delitem(sys.modules, "fastharness.runtime.openclaw", raising=False)

    return {"OpenClawClient": mock_client_cls}


class TestOpenClawRuntime:
    """Tests for OpenClawRuntime run/stream behavior."""

    @pytest.mark.asyncio
    async def test_run_returns_text(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_conv = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Hello from OpenClaw"
        mock_conv.say = AsyncMock(return_value=mock_result)

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=mock_conv, client=MagicMock())
        result = await runtime.run("Hi")

        assert result == "Hello from OpenClaw"
        mock_conv.say.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_stream_yields_text_events(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_conv = MagicMock()

        async def fake_stream(prompt):
            yield ContentEvent("streamed text")
            yield DoneEventOC(content="streamed text", token_usage=mock_usage)

        mock_conv.stream = AsyncMock(side_effect=lambda p: fake_stream(p))

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=mock_conv, client=MagicMock())
        events = []
        async for event in runtime.stream("test"):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "streamed text"
        assert isinstance(events[1], DoneEvent)
        assert events[1].final_text == "streamed text"
        assert events[1].metrics["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_stream_yields_tool_events(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_conv = MagicMock()

        async def fake_stream(prompt):
            yield ToolCallEvent("search", {"query": "test"})
            yield DoneEventOC(content="done")

        mock_conv.stream = AsyncMock(side_effect=lambda p: fake_stream(p))

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=mock_conv, client=MagicMock())
        events = []
        async for event in runtime.stream("test"):
            events.append(event)

        assert isinstance(events[0], ToolEvent)
        assert events[0].tool_name == "search"
        assert events[0].tool_input == {"query": "test"}

    @pytest.mark.asyncio
    async def test_stream_error_raises(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_conv = MagicMock()

        async def fake_stream(prompt):
            yield ErrorEvent("agent crashed")

        mock_conv.stream = AsyncMock(side_effect=lambda p: fake_stream(p))

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=mock_conv, client=MagicMock())
        with pytest.raises(RuntimeError, match="agent crashed"):
            async for _ in runtime.stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_stream_accumulates_chunks(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_conv = MagicMock()

        async def fake_stream(prompt):
            yield ContentEvent("Hello ")
            yield ContentEvent("world")
            yield DoneEventOC(content=None)  # Should use accumulated chunks

        mock_conv.stream = AsyncMock(side_effect=lambda p: fake_stream(p))

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=mock_conv, client=MagicMock())
        events = []
        async for event in runtime.stream("test"):
            events.append(event)

        done_event = events[-1]
        assert isinstance(done_event, DoneEvent)
        assert done_event.final_text == "Hello world"

    @pytest.mark.asyncio
    async def test_aclose_closes_client(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime

        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock()

        runtime = OpenClawRuntime(agent=MagicMock(), conversation=MagicMock(), client=mock_client)
        await runtime.aclose()

        mock_client.__aexit__.assert_called_once()


class TestOpenClawRuntimeFactory:
    """Tests for OpenClawRuntimeFactory lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_session(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntime, OpenClawRuntimeFactory

        mock_client = MagicMock()
        mock_agent = MagicMock()
        mock_conv = MagicMock()
        mock_agent.conversation.return_value = mock_conv
        mock_client.get_agent.return_value = mock_agent
        mock_openclaw_sdk["OpenClawClient"].connect = AsyncMock(return_value=mock_client)

        factory = OpenClawRuntimeFactory(gateway_url="ws://test:18789")
        config = _make_config()

        runtime = await factory.get_or_create("user:ctx-1", config)

        assert isinstance(runtime, OpenClawRuntime)
        mock_openclaw_sdk["OpenClawClient"].connect.assert_called_once()
        # Conversation ID should include the session key, not just agent name
        mock_agent.conversation.assert_called_once_with("fastharness:user:ctx-1")

    @pytest.mark.asyncio
    async def test_reuses_session(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntimeFactory

        mock_client = MagicMock()
        mock_agent = MagicMock()
        mock_agent.conversation.return_value = MagicMock()
        mock_client.get_agent.return_value = mock_agent
        mock_openclaw_sdk["OpenClawClient"].connect = AsyncMock(return_value=mock_client)

        factory = OpenClawRuntimeFactory(gateway_url="ws://test:18789")
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s1", config)

        # Should only connect once
        assert mock_openclaw_sdk["OpenClawClient"].connect.call_count == 1

    @pytest.mark.asyncio
    async def test_remove_session(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntimeFactory

        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.conversation.return_value = MagicMock()
        mock_client.get_agent.return_value = mock_agent
        mock_openclaw_sdk["OpenClawClient"].connect = AsyncMock(return_value=mock_client)

        factory = OpenClawRuntimeFactory(gateway_url="ws://test:18789")
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.remove("s1")

        assert len(factory._sessions) == 0

    @pytest.mark.asyncio
    async def test_shutdown_clears_all(self, mock_openclaw_sdk):
        from fastharness.runtime.openclaw import OpenClawRuntimeFactory

        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.conversation.return_value = MagicMock()
        mock_client.get_agent.return_value = mock_agent
        mock_openclaw_sdk["OpenClawClient"].connect = AsyncMock(return_value=mock_client)

        factory = OpenClawRuntimeFactory(gateway_url="ws://test:18789")
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s2", config)
        await factory.shutdown()

        assert len(factory._sessions) == 0

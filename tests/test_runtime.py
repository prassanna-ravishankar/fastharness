"""Tests for AgentRuntime protocol and Claude SDK implementation."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.core.skill import Skill
from fastharness.runtime.base import AgentRuntime, AgentRuntimeFactory
from fastharness.runtime.claude import ClaudeRuntime, ClaudeRuntimeFactory, _config_to_options


class TestConfigToOptions:
    """Tests for AgentConfig → ClaudeAgentOptions conversion."""

    def test_basic_conversion(self) -> None:
        config = AgentConfig(
            name="test",
            description="A test agent",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
            system_prompt="Be helpful",
            tools=["Read", "Write"],
            model="claude-sonnet-4-20250514",
            max_turns=5,
        )
        opts = _config_to_options(config)

        assert opts.system_prompt == "Be helpful"
        assert opts.allowed_tools == ["Read", "Write"]
        assert opts.model == "claude-sonnet-4-20250514"
        assert opts.max_turns == 5
        assert opts.permission_mode == "bypassPermissions"

    def test_none_optionals(self) -> None:
        config = AgentConfig(
            name="minimal",
            description="Minimal",
            skills=[Skill(id="s1", name="S1", description="d")],
        )
        opts = _config_to_options(config)

        assert opts.system_prompt is None
        assert opts.allowed_tools == []
        assert opts.mcp_servers == {}
        assert opts.output_format is None

    def test_mcp_servers_and_output_format(self) -> None:
        config = AgentConfig(
            name="full",
            description="Full",
            skills=[Skill(id="s1", name="S1", description="d")],
            mcp_servers={"server1": {"command": "test"}},
            output_format={"type": "json_schema", "schema": {}},
            setting_sources=["project", "user"],
        )
        opts = _config_to_options(config)

        assert opts.mcp_servers == {"server1": {"command": "test"}}
        assert opts.output_format == {"type": "json_schema", "schema": {}}


class TestClaudeRuntime:
    """Tests for ClaudeRuntime run/stream behavior."""

    def _make_assistant_msg(self, text: str) -> MagicMock:
        from claude_agent_sdk import AssistantMessage, TextBlock

        block = TextBlock(text=text)
        msg = MagicMock(spec=AssistantMessage)
        msg.content = [block]
        # Make isinstance checks work
        msg.__class__ = AssistantMessage
        return msg

    def _make_result_msg(
        self, result: str | None = None, structured_output: Any = None
    ) -> MagicMock:
        from claude_agent_sdk import ResultMessage

        msg = MagicMock(spec=ResultMessage)
        msg.result = result
        msg.structured_output = structured_output
        msg.__class__ = ResultMessage
        return msg

    def _make_tool_block(self, name: str, input_data: dict[str, Any]) -> MagicMock:
        block = MagicMock()
        block.name = name
        block.input = input_data
        # Not a TextBlock
        type(block).__name__ = "ToolUseBlock"
        return block

    @pytest.mark.asyncio
    async def test_run_returns_text(self) -> None:
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        assistant = self._make_assistant_msg("Hello world")
        result = self._make_result_msg(result="Hello world")

        async def fake_receive() -> AsyncIterator[Any]:
            yield assistant
            yield result

        mock_client.receive_response = fake_receive

        runtime = ClaudeRuntime(mock_client)
        output = await runtime.run("Hi")

        assert output == "Hello world"
        mock_client.query.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_run_returns_structured_output(self) -> None:
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        structured = {"answer": 42}
        result = self._make_result_msg(result=None, structured_output=structured)

        async def fake_receive() -> AsyncIterator[Any]:
            yield result

        mock_client.receive_response = fake_receive

        runtime = ClaudeRuntime(mock_client)
        output = await runtime.run("compute")

        assert output == {"answer": 42}

    @pytest.mark.asyncio
    async def test_run_prefers_structured_over_text(self) -> None:
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        assistant = self._make_assistant_msg("text fallback")
        structured = {"key": "value"}
        result = self._make_result_msg(result="text fallback", structured_output=structured)

        async def fake_receive() -> AsyncIterator[Any]:
            yield assistant
            yield result

        mock_client.receive_response = fake_receive

        runtime = ClaudeRuntime(mock_client)
        output = await runtime.run("test")

        assert output == {"key": "value"}

    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        assistant = self._make_assistant_msg("streaming text")
        result = self._make_result_msg(result="streaming text")

        async def fake_receive() -> AsyncIterator[Any]:
            yield assistant
            yield result

        mock_client.receive_response = fake_receive

        runtime = ClaudeRuntime(mock_client)
        events: list[Event] = []
        async for event in runtime.stream("test"):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "streaming text"
        assert isinstance(events[1], DoneEvent)
        assert events[1].final_text == "streaming text"

    @pytest.mark.asyncio
    async def test_stream_yields_tool_events(self) -> None:
        from claude_agent_sdk import AssistantMessage

        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        tool_block = self._make_tool_block("search", {"query": "test"})
        assistant = MagicMock(spec=AssistantMessage)
        assistant.content = [tool_block]
        assistant.__class__ = AssistantMessage

        result = self._make_result_msg(result="done")

        async def fake_receive() -> AsyncIterator[Any]:
            yield assistant
            yield result

        mock_client.receive_response = fake_receive

        runtime = ClaudeRuntime(mock_client)
        events: list[Event] = []
        async for event in runtime.stream("search something"):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], ToolEvent)
        assert events[0].tool_name == "search"
        assert events[0].tool_input == {"query": "test"}
        assert isinstance(events[1], DoneEvent)

    @pytest.mark.asyncio
    async def test_aclose(self) -> None:
        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock()

        runtime = ClaudeRuntime(mock_client)
        await runtime.aclose()

        mock_client.__aexit__.assert_called_once_with(None, None, None)


class TestClaudeRuntimeFactory:
    """Tests for ClaudeRuntimeFactory lifecycle management."""

    @pytest.mark.asyncio
    async def test_get_or_create_returns_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        def mock_sdk_client(options: Any) -> Any:
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        factory = ClaudeRuntimeFactory(ttl_minutes=15)
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="d")],
        )

        runtime = await factory.get_or_create("session-1", config)

        assert isinstance(runtime, ClaudeRuntime)

    @pytest.mark.asyncio
    async def test_remove_delegates_to_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        def mock_sdk_client(options: Any) -> Any:
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        factory = ClaudeRuntimeFactory(ttl_minutes=15)
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="d")],
        )

        await factory.get_or_create("session-1", config)
        await factory.remove("session-1")

        mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_delegates_to_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        def mock_sdk_client(options: Any) -> Any:
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        factory = ClaudeRuntimeFactory(ttl_minutes=15)
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="d")],
        )

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s2", config)
        await factory.shutdown()

        assert mock_client.__aexit__.call_count == 2


class TestBaseSessionFactory:
    """Tests for the shared BaseSessionFactory."""

    def _make_factory(self) -> Any:
        import logging

        from fastharness.runtime.base import BaseSessionFactory, SessionEntry

        class StubSession(SessionEntry):
            value: str = "stub"

        class StubFactory(BaseSessionFactory):
            create_count: int = 0

            async def _create_session(self, config: AgentConfig) -> StubSession:
                self.create_count += 1
                return StubSession()

            def _build_runtime(self, entry: SessionEntry) -> Any:
                return MagicMock()

        return StubFactory(ttl_minutes=15, logger=logging.getLogger("test"))

    @pytest.mark.asyncio
    async def test_creates_session(self) -> None:
        factory = self._make_factory()
        config = AgentConfig(
            name="t", description="t", skills=[Skill(id="s", name="S", description="d")]
        )
        runtime = await factory.get_or_create("k1", config)
        assert runtime is not None
        assert len(factory._sessions) == 1

    @pytest.mark.asyncio
    async def test_reuses_session(self) -> None:
        factory = self._make_factory()
        config = AgentConfig(
            name="t", description="t", skills=[Skill(id="s", name="S", description="d")]
        )
        await factory.get_or_create("k1", config)
        await factory.get_or_create("k1", config)
        assert factory.create_count == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_remove(self) -> None:
        factory = self._make_factory()
        config = AgentConfig(
            name="t", description="t", skills=[Skill(id="s", name="S", description="d")]
        )
        await factory.get_or_create("k1", config)
        await factory.remove("k1")
        assert len(factory._sessions) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_noop(self) -> None:
        factory = self._make_factory()
        await factory.remove("doesnt-exist")  # should not raise

    @pytest.mark.asyncio
    async def test_shutdown_clears_all(self) -> None:
        factory = self._make_factory()
        config = AgentConfig(
            name="t", description="t", skills=[Skill(id="s", name="S", description="d")]
        )
        await factory.get_or_create("k1", config)
        await factory.get_or_create("k2", config)
        await factory.shutdown()
        assert len(factory._sessions) == 0

    def test_session_entry_stale(self) -> None:
        from fastharness.runtime.base import SessionEntry

        entry = SessionEntry()
        assert not entry.is_stale(15)  # just created
        assert entry.is_stale(0)  # 0 TTL → always stale

    def test_ttl_validation_rejects_zero(self) -> None:
        import logging

        from fastharness.runtime.base import BaseSessionFactory

        with pytest.raises(ValueError, match="ttl_minutes must be >= 1"):
            BaseSessionFactory(ttl_minutes=0, logger=logging.getLogger("test"))

    def test_ttl_validation_rejects_negative(self) -> None:
        import logging

        from fastharness.runtime.base import BaseSessionFactory

        with pytest.raises(ValueError, match="ttl_minutes must be >= 1"):
            BaseSessionFactory(ttl_minutes=-5, logger=logging.getLogger("test"))

    def test_session_entry_touch(self) -> None:
        from datetime import timedelta

        from fastharness.runtime.base import SessionEntry

        entry = SessionEntry()
        old_time = entry.last_accessed
        entry.last_accessed -= timedelta(minutes=5)
        entry.touch()
        assert entry.last_accessed >= old_time


class TestProtocolConformance:
    """Verify ClaudeRuntime and ClaudeRuntimeFactory satisfy their protocols."""

    def test_claude_runtime_is_agent_runtime(self) -> None:
        mock_client = MagicMock()
        runtime = ClaudeRuntime(mock_client)
        assert isinstance(runtime, AgentRuntime)

    def test_claude_runtime_factory_is_agent_runtime_factory(self) -> None:
        factory = ClaudeRuntimeFactory()
        assert isinstance(factory, AgentRuntimeFactory)


class TestHarnessClientRuntimeIntegration:
    """Tests for HarnessClient using runtime field."""

    @pytest.mark.asyncio
    async def test_run_delegates_to_runtime(self) -> None:
        async def fake_stream(prompt: str) -> AsyncIterator[Event]:
            yield TextEvent(text="runtime result")
            yield DoneEvent(final_text="runtime result")

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = fake_stream

        from fastharness import HarnessClient

        client = HarnessClient(runtime=mock_runtime)
        result = await client.run("test prompt")

        assert result == "runtime result"

    @pytest.mark.asyncio
    async def test_run_structured_output_via_runtime(self) -> None:
        async def fake_stream(prompt: str) -> AsyncIterator[Event]:
            yield DoneEvent(final_text=None, structured_output={"structured": True})

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = fake_stream

        from fastharness import HarnessClient

        client = HarnessClient(runtime=mock_runtime)
        result = await client.run("test")

        assert result == {"structured": True}

    @pytest.mark.asyncio
    async def test_stream_delegates_to_runtime(self) -> None:
        async def fake_stream(prompt: str) -> AsyncIterator[Event]:
            yield TextEvent(text="hello")
            yield DoneEvent(final_text="hello")

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = fake_stream

        from fastharness import HarnessClient

        client = HarnessClient(runtime=mock_runtime)
        events: list[Event] = []
        async for event in client.stream("test"):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert isinstance(events[1], DoneEvent)

    @pytest.mark.asyncio
    async def test_run_wraps_runtime_error(self) -> None:
        async def failing_stream(prompt: str) -> AsyncIterator[Event]:
            raise ValueError("boom")
            yield  # type: ignore[misc]

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = failing_stream

        from fastharness import HarnessClient

        client = HarnessClient(runtime=mock_runtime)
        with pytest.raises(RuntimeError, match="Agent.*failed.*boom"):
            await client.run("test")

    @pytest.mark.asyncio
    async def test_stream_wraps_runtime_error(self) -> None:
        async def failing_stream(prompt: str) -> AsyncIterator[Event]:
            raise ConnectionError("network down")
            yield  # type: ignore[misc]  # make it a generator

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = failing_stream

        from fastharness import HarnessClient

        client = HarnessClient(runtime=mock_runtime)
        with pytest.raises(RuntimeError, match="Agent streaming failed.*network down"):
            async for _ in client.stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_run_step_logging_with_runtime(self) -> None:
        """Verify step logger fires when using a runtime (via stream delegation)."""

        async def fake_stream(prompt: str) -> AsyncIterator[Event]:
            yield TextEvent(text="logged result")
            yield DoneEvent(final_text="logged result")

        mock_runtime = MagicMock(spec=AgentRuntime)
        mock_runtime.stream = fake_stream

        from fastharness import HarnessClient

        mock_logger = AsyncMock()

        client = HarnessClient(runtime=mock_runtime, step_logger=mock_logger)
        await client.run("test")

        # Should have logged assistant_message + turn_complete (from stream path)
        step_types = [c[0][0].step_type for c in mock_logger.log_step.call_args_list]
        assert "assistant_message" in step_types
        assert "turn_complete" in step_types

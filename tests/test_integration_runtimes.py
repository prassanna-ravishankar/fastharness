"""Integration tests for runtime backends — real API calls, no mocks.

Requires:
  - ANTHROPIC_API_KEY in .env or environment
  - Optional deps: pip install fastharness[openhands,deepagents]

Run with: uv run pytest -m integration tests/test_integration_runtimes.py -v
"""

import os

import pytest

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, TextEvent
from fastharness.core.skill import Skill

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SIMPLE_CONFIG = AgentConfig(
    name="integration-test",
    description="Integration test agent",
    skills=[Skill(id="echo", name="Echo", description="Echo back")],
    system_prompt="Respond with exactly one word: PONG",
    max_turns=1,
)


def has_api_key() -> bool:
    import fastharness  # noqa: F401 — triggers .env loading

    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_no_key = pytest.mark.skipif(not has_api_key(), reason="ANTHROPIC_API_KEY not set")


# ===========================================================================
# Claude runtime
# ===========================================================================


class TestClaudeRuntimeIntegration:
    """Live tests for ClaudeRuntime via Claude Agent SDK subprocess."""

    @pytest.mark.asyncio
    @skip_no_key
    async def test_run_returns_text(self) -> None:
        from fastharness.runtime.claude import ClaudeRuntimeFactory

        factory = ClaudeRuntimeFactory(ttl_minutes=1)
        try:
            runtime = await factory.get_or_create("claude-run", SIMPLE_CONFIG)
            result = await runtime.run("Ping")

            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_stream_yields_events(self) -> None:
        from fastharness.runtime.claude import ClaudeRuntimeFactory

        factory = ClaudeRuntimeFactory(ttl_minutes=1)
        try:
            runtime = await factory.get_or_create("claude-stream", SIMPLE_CONFIG)
            events = [e async for e in runtime.stream("Ping")]

            text_events = [e for e in events if isinstance(e, TextEvent)]
            done_events = [e for e in events if isinstance(e, DoneEvent)]

            assert len(text_events) >= 1, "Expected at least one TextEvent"
            assert len(done_events) == 1, "Expected exactly one DoneEvent"
            assert done_events[0].final_text is not None
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_factory_session_reuse(self) -> None:
        from fastharness.runtime.claude import ClaudeRuntimeFactory

        factory = ClaudeRuntimeFactory(ttl_minutes=1)
        try:
            r1 = await factory.get_or_create("claude-reuse", SIMPLE_CONFIG)
            r2 = await factory.get_or_create("claude-reuse", SIMPLE_CONFIG)

            # Same session key → same underlying pool entry
            assert r1._client is r2._client
        finally:
            await factory.shutdown()


# ===========================================================================
# OpenHands runtime
# ===========================================================================


class TestOpenHandsRuntimeIntegration:
    """Live tests for OpenHandsRuntime via OpenHands SDK."""

    @pytest.mark.asyncio
    @skip_no_key
    async def test_run_returns_text(self) -> None:
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        config = AgentConfig(
            name="oh-test",
            description="OpenHands test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="anthropic/claude-sonnet-4-5-20250929",
            max_turns=1,
        )

        factory = OpenHandsRuntimeFactory(ttl_minutes=1, workspace="/tmp/oh_integ")
        try:
            runtime = await factory.get_or_create("oh-run", config)
            result = await runtime.run("Say hello in one sentence")

            assert isinstance(result, str)
            assert len(result) > 0, "Expected non-empty response"
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_stream_yields_events(self) -> None:
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        config = AgentConfig(
            name="oh-test",
            description="OpenHands test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="anthropic/claude-sonnet-4-5-20250929",
            max_turns=1,
        )

        factory = OpenHandsRuntimeFactory(ttl_minutes=1, workspace="/tmp/oh_integ_stream")
        try:
            runtime = await factory.get_or_create("oh-stream", config)
            events = [e async for e in runtime.stream("Say hello in one sentence")]

            text_events = [e for e in events if isinstance(e, TextEvent)]
            done_events = [e for e in events if isinstance(e, DoneEvent)]

            assert len(text_events) >= 1, "Expected at least one TextEvent"
            assert len(done_events) == 1, "Expected exactly one DoneEvent"
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_factory_session_reuse(self) -> None:
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        config = AgentConfig(
            name="oh-test",
            description="OpenHands test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="anthropic/claude-sonnet-4-5-20250929",
            max_turns=1,
        )

        factory = OpenHandsRuntimeFactory(ttl_minutes=1, workspace="/tmp/oh_integ_reuse")
        try:
            await factory.get_or_create("oh-reuse", config)
            await factory.get_or_create("oh-reuse", config)

            # Should only have one session entry
            assert len(factory._sessions) == 1
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_factory_remove(self) -> None:
        from fastharness.runtime.openhands import OpenHandsRuntimeFactory

        config = AgentConfig(
            name="oh-test",
            description="OpenHands test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="anthropic/claude-sonnet-4-5-20250929",
            max_turns=1,
        )

        factory = OpenHandsRuntimeFactory(ttl_minutes=1, workspace="/tmp/oh_integ_rm")
        try:
            await factory.get_or_create("oh-rm", config)
            assert len(factory._sessions) == 1

            await factory.remove("oh-rm")
            assert len(factory._sessions) == 0
        finally:
            await factory.shutdown()


# ===========================================================================
# DeepAgents runtime
# ===========================================================================


class TestDeepAgentsRuntimeIntegration:
    """Live tests for DeepAgentsRuntime via Pydantic DeepAgents."""

    @pytest.mark.asyncio
    @skip_no_key
    async def test_run_returns_text(self) -> None:
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        config = AgentConfig(
            name="deep-test",
            description="DeepAgents test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            system_prompt="Respond with exactly one word: PONG",
            model="claude-sonnet-4-5-20250929",
        )

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        try:
            runtime = await factory.get_or_create("deep-run", config)
            result = await runtime.run("Ping")

            assert isinstance(result, str)
            assert len(result) > 0, "Expected non-empty response"
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_stream_yields_events(self) -> None:
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        config = AgentConfig(
            name="deep-test",
            description="DeepAgents test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            system_prompt="Respond with exactly one word: PONG",
            model="claude-sonnet-4-5-20250929",
        )

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        try:
            runtime = await factory.get_or_create("deep-stream", config)
            events = [e async for e in runtime.stream("Ping")]

            done_events = [e for e in events if isinstance(e, DoneEvent)]
            assert len(done_events) == 1, "Expected exactly one DoneEvent"
            # Stream should yield at least a DoneEvent; TextEvent depends on output
            assert len(events) >= 1
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_multi_turn_history(self) -> None:
        """Verify message history is preserved across turns."""
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        config = AgentConfig(
            name="deep-test",
            description="DeepAgents test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            system_prompt=(
                "You are a memory test agent. When asked to remember something, "
                "confirm you stored it. When asked to recall, repeat it exactly."
            ),
            model="claude-sonnet-4-5-20250929",
        )

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        try:
            runtime = await factory.get_or_create("deep-multi", config)

            # Turn 1: store a fact
            r1 = await runtime.run("Remember the code word: ORANGE")
            assert isinstance(r1, str)
            assert len(r1) > 0

            # Turn 2: recall it — history should contain the previous exchange
            r2 = await runtime.run("What was the code word I told you?")
            assert isinstance(r2, str)
            assert "ORANGE" in r2.upper(), f"Expected 'ORANGE' in recall, got: {r2}"
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_factory_session_reuse(self) -> None:
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        config = AgentConfig(
            name="deep-test",
            description="DeepAgents test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="claude-sonnet-4-5-20250929",
        )

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        try:
            await factory.get_or_create("deep-reuse", config)
            await factory.get_or_create("deep-reuse", config)

            assert len(factory._sessions) == 1
        finally:
            await factory.shutdown()

    @pytest.mark.asyncio
    @skip_no_key
    async def test_factory_remove(self) -> None:
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        config = AgentConfig(
            name="deep-test",
            description="DeepAgents test",
            skills=[Skill(id="echo", name="Echo", description="Echo back")],
            model="claude-sonnet-4-5-20250929",
        )

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        try:
            await factory.get_or_create("deep-rm", config)
            assert len(factory._sessions) == 1

            await factory.remove("deep-rm")
            assert len(factory._sessions) == 0
        finally:
            await factory.shutdown()

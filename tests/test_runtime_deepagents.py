"""Tests for Pydantic DeepAgents runtime implementation (mocked — no SDK dependency needed)."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastharness.core.agent import AgentConfig
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
def mock_deepagents_sdk(monkeypatch):
    """Inject fake pydantic_deep / pydantic_ai_backends modules."""
    # pydantic_ai and submodules
    pydantic_ai_mod = ModuleType("pydantic_ai")
    pydantic_ai_graph_mod = ModuleType("pydantic_ai._agent_graph")
    mock_call_tools_node = type("CallToolsNode", (), {})
    pydantic_ai_graph_mod.CallToolsNode = mock_call_tools_node

    # pydantic_ai_backends
    backends_mod = ModuleType("pydantic_ai_backends")
    mock_state_backend = MagicMock(name="StateBackend")
    backends_mod.StateBackend = mock_state_backend

    # pydantic_deep
    deep_mod = ModuleType("pydantic_deep")
    mock_create_deep_agent = MagicMock(name="create_deep_agent")
    mock_deep_agent_deps = MagicMock(name="DeepAgentDeps")
    deep_mod.create_deep_agent = mock_create_deep_agent
    deep_mod.DeepAgentDeps = mock_deep_agent_deps

    monkeypatch.setitem(sys.modules, "pydantic_ai", pydantic_ai_mod)
    monkeypatch.setitem(sys.modules, "pydantic_ai._agent_graph", pydantic_ai_graph_mod)
    monkeypatch.setitem(sys.modules, "pydantic_ai_backends", backends_mod)
    monkeypatch.setitem(sys.modules, "pydantic_deep", deep_mod)

    # Clear cached import
    monkeypatch.delitem(sys.modules, "fastharness.runtime.deepagents", raising=False)

    return {
        "create_deep_agent": mock_create_deep_agent,
        "DeepAgentDeps": mock_deep_agent_deps,
        "StateBackend": mock_state_backend,
        "CallToolsNode": mock_call_tools_node,
    }


class TestDeepAgentsRuntime:
    """Tests for DeepAgentsRuntime run/stream behavior."""

    @pytest.mark.asyncio
    async def test_run_returns_text(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntime

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "Deep agent response"
        mock_result.all_messages.return_value = ["msg1"]
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_deps = MagicMock()

        runtime = DeepAgentsRuntime(mock_agent, mock_deps, [])
        result = await runtime.run("test prompt")

        assert result == "Deep agent response"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_preserves_message_history(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntime

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "first"
        mock_result.all_messages.return_value = ["msg1"]
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_deps = MagicMock()
        history: list = []

        runtime = DeepAgentsRuntime(mock_agent, mock_deps, history)
        await runtime.run("first prompt")

        # History should be updated in-place
        assert history == ["msg1"]

        # Second call should pass message_history kwarg
        mock_result2 = MagicMock()
        mock_result2.output = "second"
        mock_result2.all_messages.return_value = ["msg1", "msg2"]
        mock_agent.run = AsyncMock(return_value=mock_result2)

        await runtime.run("second prompt")

        # Verify message_history was passed and history updated
        call_kwargs = mock_agent.run.call_args[1]
        assert "message_history" in call_kwargs
        assert history == ["msg1", "msg2"]

    @pytest.mark.asyncio
    async def test_aclose_clears_history(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntime

        history = ["msg1", "msg2"]
        runtime = DeepAgentsRuntime(MagicMock(), MagicMock(), history)
        await runtime.aclose()

        assert history == []


class TestDeepAgentsRuntimeFactory:
    """Tests for DeepAgentsRuntimeFactory lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_new_session(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntime, DeepAgentsRuntimeFactory

        factory = DeepAgentsRuntimeFactory(ttl_minutes=15)
        config = _make_config(system_prompt="Be helpful", model="openai:gpt-4.1")

        runtime = await factory.get_or_create("s1", config)

        assert isinstance(runtime, DeepAgentsRuntime)
        mock_deepagents_sdk["create_deep_agent"].assert_called_once_with(
            include_subagents=False,
            include_skills=False,
            include_web=False,
            include_filesystem=False,
            include_todo=False,
            model="openai:gpt-4.1",
            instructions="Be helpful",
        )

    @pytest.mark.asyncio
    async def test_reuses_existing_session(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        factory = DeepAgentsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s1", config)

        # Agent should only be created once
        assert mock_deepagents_sdk["create_deep_agent"].call_count == 1

    @pytest.mark.asyncio
    async def test_remove_session(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        factory = DeepAgentsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.remove("s1")

        assert "s1" not in factory._sessions

    @pytest.mark.asyncio
    async def test_shutdown_clears_all(self, mock_deepagents_sdk):
        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        factory = DeepAgentsRuntimeFactory(ttl_minutes=15)
        config = _make_config()

        await factory.get_or_create("s1", config)
        await factory.get_or_create("s2", config)
        await factory.shutdown()

        assert len(factory._sessions) == 0

    @pytest.mark.asyncio
    async def test_stale_cleanup(self, mock_deepagents_sdk):
        from datetime import timedelta

        from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

        factory = DeepAgentsRuntimeFactory(ttl_minutes=1)
        config = _make_config()

        await factory.get_or_create("s1", config)

        # Backdate to make stale
        factory._sessions["s1"].last_accessed -= timedelta(minutes=2)

        async with factory._lock:
            stale = [k for k, v in factory._sessions.items() if v.is_stale(1)]
            for key in stale:
                factory._sessions.pop(key)

        assert "s1" not in factory._sessions

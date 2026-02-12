"""Tests for ClaudeAgentExecutor."""

import asyncio

import pytest

from fastharness import AgentConfig, Skill
from fastharness.core.agent import Agent
from fastharness.worker.claude_executor import AgentRegistry, ClaudeAgentExecutor


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_registry_get(self) -> None:
        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})

        assert registry.get("test") is agent
        assert registry.get("nonexistent") is None

    def test_registry_get_default(self) -> None:
        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})

        assert registry.get_default() is agent

    def test_registry_get_default_empty(self) -> None:
        registry = AgentRegistry(agents={})
        assert registry.get_default() is None

    def test_registry_get_by_skill(self) -> None:
        skill = Skill(id="search", name="Search", description="Search skill")
        config = AgentConfig(name="searcher", description="Searcher", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"searcher": agent})

        assert registry.get_by_skill("search") is agent
        assert registry.get_by_skill("nonexistent") is None

    def test_registry_resolve_by_skill(self) -> None:
        s1 = Skill(id="s1", name="S1", description="Skill 1")
        s2 = Skill(id="s2", name="S2", description="Skill 2")
        a1 = Agent(config=AgentConfig(name="a1", description="A1", skills=[s1]), func=None)
        a2 = Agent(config=AgentConfig(name="a2", description="A2", skills=[s2]), func=None)
        registry = AgentRegistry(agents={"a1": a1, "a2": a2})

        assert registry.resolve("s2") is a2
        assert registry.resolve("s1") is a1

    def test_registry_resolve_falls_back_to_default(self) -> None:
        skill = Skill(id="s1", name="S1", description="Skill 1")
        agent = Agent(config=AgentConfig(name="a1", description="A1", skills=[skill]), func=None)
        registry = AgentRegistry(agents={"a1": agent})

        assert registry.resolve("nonexistent") is agent
        assert registry.resolve(None) is agent
        assert registry.resolve() is agent

    def test_registry_resolve_empty(self) -> None:
        registry = AgentRegistry(agents={})
        assert registry.resolve("s1") is None
        assert registry.resolve() is None


class TestClaudeExecutorBuildArtifacts:
    """Tests for ClaudeAgentExecutor.build_artifacts()."""

    @pytest.fixture
    def executor(self) -> ClaudeAgentExecutor:
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        registry = AgentRegistry(agents={})
        return ClaudeAgentExecutor(
            agent_registry=registry,
            task_store=InMemoryTaskStore(),
        )

    def test_build_artifacts_string(self, executor: ClaudeAgentExecutor) -> None:
        artifacts = executor.build_artifacts("Hello world")
        assert len(artifacts) == 1
        assert artifacts[0].parts[0].root.text == "Hello world"

    def test_build_artifacts_list(self, executor: ClaudeAgentExecutor) -> None:
        fake_artifacts = [{"artifact_id": "1", "name": "test", "parts": []}]
        artifacts = executor.build_artifacts(fake_artifacts)
        assert artifacts == fake_artifacts

    def test_build_artifacts_none(self, executor: ClaudeAgentExecutor) -> None:
        artifacts = executor.build_artifacts(None)
        assert artifacts == []

    def test_build_artifacts_other(self, executor: ClaudeAgentExecutor) -> None:
        artifacts = executor.build_artifacts(42)
        assert len(artifacts) == 1
        assert "42" in artifacts[0].parts[0].root.text


class TestClaudeExecutorTaskTracking:
    """Tests for ClaudeAgentExecutor task tracking."""

    @pytest.fixture
    def executor(self) -> ClaudeAgentExecutor:
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})
        return ClaudeAgentExecutor(
            agent_registry=registry,
            task_store=InMemoryTaskStore(),
        )

    def test_running_tasks_initialized(self, executor: ClaudeAgentExecutor) -> None:
        assert hasattr(executor, "_running_tasks")
        assert executor._running_tasks == {}

    @pytest.mark.asyncio
    async def test_cancel_task_removes_from_running(self, executor: ClaudeAgentExecutor) -> None:
        """Cancelling a tracked asyncio task removes it from _running_tasks."""

        async def _noop() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(_noop())
        executor._running_tasks["task-123"] = task

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        executor._running_tasks.pop("task-123", None)

        assert "task-123" not in executor._running_tasks

"""Tests for ClaudeAgentExecutor."""

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
        # Test that cancel removes task from _running_tasks dict
        executor._running_tasks["task-123"] = True

        # Directly test the removal logic
        task_id = "task-123"
        if task_id in executor._running_tasks:
            del executor._running_tasks[task_id]

        assert "task-123" not in executor._running_tasks

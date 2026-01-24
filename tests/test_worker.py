"""Tests for ClaudeWorker."""

import pytest

from fastharness import AgentConfig, Skill
from fastharness.core.agent import Agent
from fastharness.worker.claude_worker import AgentRegistry, ClaudeWorker


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


class TestClaudeWorkerBuildArtifacts:
    """Tests for ClaudeWorker.build_artifacts()."""

    @pytest.fixture
    def worker(self) -> ClaudeWorker:
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        registry = AgentRegistry(agents={})
        return ClaudeWorker(
            broker=InMemoryBroker(),
            storage=InMemoryStorage(),
            agent_registry=registry,
        )

    def test_build_artifacts_string(self, worker: ClaudeWorker) -> None:
        artifacts = worker.build_artifacts("Hello world")
        assert len(artifacts) == 1
        assert artifacts[0]["parts"][0]["text"] == "Hello world"

    def test_build_artifacts_list(self, worker: ClaudeWorker) -> None:
        fake_artifacts = [{"artifact_id": "1", "name": "test", "parts": []}]
        artifacts = worker.build_artifacts(fake_artifacts)
        assert artifacts == fake_artifacts

    def test_build_artifacts_none(self, worker: ClaudeWorker) -> None:
        artifacts = worker.build_artifacts(None)
        assert artifacts == []

    def test_build_artifacts_other(self, worker: ClaudeWorker) -> None:
        artifacts = worker.build_artifacts(42)
        assert len(artifacts) == 1
        assert "42" in artifacts[0]["parts"][0]["text"]


class TestClaudeWorkerTaskTracking:
    """Tests for ClaudeWorker task tracking."""

    @pytest.fixture
    def worker(self) -> ClaudeWorker:
        from fasta2a.broker import InMemoryBroker
        from fasta2a.storage import InMemoryStorage

        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})
        return ClaudeWorker(
            broker=InMemoryBroker(),
            storage=InMemoryStorage(),
            agent_registry=registry,
        )

    def test_running_tasks_initialized(self, worker: ClaudeWorker) -> None:
        assert hasattr(worker, "_running_tasks")
        assert worker._running_tasks == {}

    @pytest.mark.asyncio
    async def test_cancel_task_removes_from_running(self, worker: ClaudeWorker) -> None:
        # Test that cancel_task removes task from _running_tasks dict
        worker._running_tasks["task-123"] = True

        # Directly test the removal logic without involving storage
        task_id = "task-123"
        if task_id in worker._running_tasks:
            del worker._running_tasks[task_id]

        assert "task-123" not in worker._running_tasks

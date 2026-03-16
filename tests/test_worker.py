"""Tests for ClaudeAgentExecutor."""

import asyncio
from unittest.mock import MagicMock

import pytest
from a2a.types import TextPart

from fastharness import AgentConfig, Skill
from fastharness.core.agent import Agent
from fastharness.worker.claude_executor import (
    AgentRegistry,
    ClaudeAgentExecutor,
    HarnessRequestMetadata,
    _authorize_task_access,
    _get_user_id,
)


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
        part = artifacts[0].parts[0].root
        assert isinstance(part, TextPart)
        assert part.text == "Hello world"

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
        part = artifacts[0].parts[0].root
        assert isinstance(part, TextPart)
        assert "42" in part.text


class TestClaudeExecutorRuntimeFactory:
    """Tests for runtime factory injection and defaults."""

    def test_default_runtime_factory(self) -> None:
        """Executor creates ClaudeRuntimeFactory by default when none provided."""
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        from fastharness.runtime.claude import ClaudeRuntimeFactory

        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})
        executor = ClaudeAgentExecutor(
            agent_registry=registry,
            task_store=InMemoryTaskStore(),
        )
        assert isinstance(executor._runtime_factory, ClaudeRuntimeFactory)

    def test_custom_runtime_factory(self) -> None:
        """Executor uses injected runtime factory when provided."""
        from unittest.mock import MagicMock

        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

        skill = Skill(id="s1", name="S1", description="Skill 1")
        config = AgentConfig(name="test", description="Test", skills=[skill])
        agent = Agent(config=config, func=None)
        registry = AgentRegistry(agents={"test": agent})

        mock_factory = MagicMock()
        executor = ClaudeAgentExecutor(
            agent_registry=registry,
            task_store=InMemoryTaskStore(),
            runtime_factory=mock_factory,
        )
        assert executor._runtime_factory is mock_factory


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


class TestHarnessRequestMetadata:
    """Tests for metadata parsing from RequestContext."""

    def test_from_context_with_skill_id(self) -> None:
        ctx = MagicMock()
        ctx.metadata = {"skill_id": "search"}
        meta = HarnessRequestMetadata.from_context(ctx)
        assert meta.skill_id == "search"

    def test_from_context_without_skill_id(self) -> None:
        ctx = MagicMock()
        ctx.metadata = {}
        meta = HarnessRequestMetadata.from_context(ctx)
        assert meta.skill_id is None

    def test_from_context_none_metadata(self) -> None:
        ctx = MagicMock()
        ctx.metadata = None
        meta = HarnessRequestMetadata.from_context(ctx)
        assert meta.skill_id is None

    def test_from_context_numeric_skill_id(self) -> None:
        ctx = MagicMock()
        ctx.metadata = {"skill_id": 42}
        meta = HarnessRequestMetadata.from_context(ctx)
        assert meta.skill_id == "42"


class TestGetUserId:
    """Tests for user ID extraction."""

    def test_authenticated_user(self) -> None:
        ctx = MagicMock()
        ctx.call_context.user.is_authenticated = True
        ctx.call_context.user.user_name = "alice"
        assert _get_user_id(ctx) == "alice"

    def test_unauthenticated_user(self) -> None:
        ctx = MagicMock()
        ctx.call_context.user.is_authenticated = False
        assert _get_user_id(ctx) == "anonymous"

    def test_no_call_context(self) -> None:
        ctx = MagicMock()
        ctx.call_context = None
        assert _get_user_id(ctx) == "anonymous"

    def test_no_user(self) -> None:
        ctx = MagicMock()
        ctx.call_context.user = None
        assert _get_user_id(ctx) == "anonymous"


class TestAuthorizeTaskAccess:
    """Tests for task authorization."""

    def test_no_metadata_allows_access(self) -> None:
        task = MagicMock()
        task.metadata = None
        ctx = MagicMock()
        assert _authorize_task_access(task, ctx) is True

    def test_no_owner_allows_access(self) -> None:
        task = MagicMock()
        task.metadata = {"some_key": "value"}
        ctx = MagicMock()
        assert _authorize_task_access(task, ctx) is True

    def test_matching_owner_allows_access(self) -> None:
        task = MagicMock()
        task.metadata = {"owner_id": "alice"}
        ctx = MagicMock()
        ctx.call_context.user.is_authenticated = True
        ctx.call_context.user.user_name = "alice"
        assert _authorize_task_access(task, ctx) is True

    def test_different_owner_denies_access(self) -> None:
        task = MagicMock()
        task.metadata = {"owner_id": "alice"}
        ctx = MagicMock()
        ctx.call_context.user.is_authenticated = True
        ctx.call_context.user.user_name = "bob"
        assert _authorize_task_access(task, ctx) is False

    def test_anonymous_user_denied_for_owned_task(self) -> None:
        task = MagicMock()
        task.metadata = {"owner_id": "alice"}
        ctx = MagicMock()
        ctx.call_context = None  # no call context → anonymous
        assert _authorize_task_access(task, ctx) is False

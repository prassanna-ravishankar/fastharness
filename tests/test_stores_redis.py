"""Tests for RedisTaskStore."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus, TextPart


def _make_task(task_id: str = "task-1", context_id: str = "ctx-1") -> Task:
    """Create a minimal Task for testing."""
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.completed),
        history=[
            Message(
                role=Role.user,
                message_id="msg-1",
                parts=[Part(root=TextPart(kind="text", text="Hello"))],
            ),
            Message(
                role=Role.agent,
                message_id="msg-2",
                parts=[Part(root=TextPart(kind="text", text="Hi there!"))],
            ),
        ],
        artifacts=[],
    )


@pytest.fixture
def mock_redis(monkeypatch):
    """Inject a mock redis.asyncio module."""
    redis_mod = ModuleType("redis")
    aioredis_mod = ModuleType("redis.asyncio")

    mock_client = MagicMock()
    mock_client.setex = AsyncMock()
    mock_client.set = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.getex = AsyncMock(return_value=None)
    mock_client.delete = AsyncMock()
    mock_client.aclose = AsyncMock()

    aioredis_mod.from_url = MagicMock(return_value=mock_client)

    monkeypatch.setitem(sys.modules, "redis", redis_mod)
    monkeypatch.setitem(sys.modules, "redis.asyncio", aioredis_mod)
    monkeypatch.delitem(sys.modules, "fastharness.stores.redis", raising=False)

    return mock_client


class TestRedisTaskStore:
    """Unit tests for RedisTaskStore with mocked Redis."""

    @pytest.mark.asyncio
    async def test_save_task(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379", ttl_seconds=3600)
        task = _make_task()

        await store.save(task)

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "fastharness:task:task-1"
        assert args[1] == 3600
        assert "task-1" in args[2]

    @pytest.mark.asyncio
    async def test_save_without_ttl(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379", ttl_seconds=0)
        task = _make_task()

        await store.save(task)

        mock_redis.set.assert_called_once()
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_uses_getex_with_ttl(self, mock_redis):
        """get() should use GETEX to atomically fetch and refresh TTL."""
        from fastharness.stores.redis import RedisTaskStore

        task = _make_task()
        mock_redis.getex = AsyncMock(return_value=task.model_dump_json())

        store = RedisTaskStore(url="redis://test:6379", ttl_seconds=3600)
        result = await store.get("task-1")

        assert result is not None
        assert result.id == "task-1"
        mock_redis.getex.assert_called_once_with("fastharness:task:task-1", ex=3600)
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_uses_plain_get_without_ttl(self, mock_redis):
        """get() with ttl_seconds=0 should use plain GET (no TTL refresh)."""
        from fastharness.stores.redis import RedisTaskStore

        task = _make_task()
        mock_redis.get = AsyncMock(return_value=task.model_dump_json())

        store = RedisTaskStore(url="redis://test:6379", ttl_seconds=0)
        result = await store.get("task-1")

        assert result is not None
        mock_redis.get.assert_called_once_with("fastharness:task:task-1")
        mock_redis.getex.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379")
        result = await store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379")
        await store.delete("task-1")

        mock_redis.delete.assert_called_once_with("fastharness:task:task-1")

    @pytest.mark.asyncio
    async def test_close(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379")
        await store.close()

        mock_redis.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        async with RedisTaskStore(url="redis://test:6379") as store:
            assert store is not None
        mock_redis.aclose.assert_called_once()

    def test_ttl_validation(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        with pytest.raises(ValueError, match="ttl_seconds must be >= 0"):
            RedisTaskStore(url="redis://test:6379", ttl_seconds=-1)

    def test_custom_prefix(self, mock_redis):
        from fastharness.stores.redis import RedisTaskStore

        store = RedisTaskStore(url="redis://test:6379", key_prefix="myapp:")
        assert store._key("task-1") == "myapp:task-1"

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_history(self, mock_redis):
        """Verify task with full conversation history survives serialization."""
        from fastharness.stores.redis import RedisTaskStore

        task = _make_task()
        stored_json = None

        async def capture_setex(key, ttl, data):
            nonlocal stored_json
            stored_json = data

        mock_redis.setex = AsyncMock(side_effect=capture_setex)
        mock_redis.getex = AsyncMock(side_effect=lambda k, **kw: stored_json)

        store = RedisTaskStore(url="redis://test:6379")

        await store.save(task)
        recovered = await store.get("task-1")

        assert recovered is not None
        assert recovered.id == task.id
        assert recovered.context_id == task.context_id
        assert len(recovered.history) == 2
        assert recovered.history[0].role == Role.user
        assert recovered.history[1].role == Role.agent
        assert recovered.status.state == TaskState.completed

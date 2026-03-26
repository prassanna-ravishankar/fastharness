"""Redis-backed TaskStore for distributed FastHarness deployments.

Stores A2A Task objects in Redis with automatic TTL expiration.
Enables multi-turn conversations across pod restarts in k8s.

Usage::

    from fastharness import FastHarness
    from fastharness.stores.redis import RedisTaskStore

    store = RedisTaskStore("redis://localhost:6379")
    harness = FastHarness(name="my-agent", task_store=store)

Requires: pip install fastharness[redis]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from a2a.server.tasks.task_store import TaskStore
from a2a.types import Task

from fastharness.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext

logger = get_logger("stores.redis")

_KEY_PREFIX = "fastharness:task:"


class RedisTaskStore(TaskStore):
    """TaskStore backed by Redis with TTL-based expiration.

    Tasks are serialized as JSON via Pydantic's model_dump_json/model_validate_json.
    Each task key is prefixed with ``fastharness:task:`` to avoid collisions.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
        key_prefix: str = _KEY_PREFIX,
        **redis_kwargs: Any,
    ) -> None:
        """Initialize the Redis task store.

        Args:
            url: Redis connection URL.
            ttl_seconds: Time-to-live for task keys (default 1 hour).
                Set to 0 to disable expiration.
            key_prefix: Key prefix for task storage.
            **redis_kwargs: Extra kwargs passed to redis.asyncio.from_url()
                (e.g. max_connections for pool sizing).
        """
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "Redis is required for RedisTaskStore. "
                "Install with: pip install fastharness[redis]"
            ) from e

        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")
        self._client = aioredis.from_url(url, **redis_kwargs)
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        """Save a task to Redis as JSON with TTL."""
        key = self._key(task.id)
        data = task.model_dump_json()
        if self._ttl > 0:
            await self._client.setex(key, self._ttl, data)
        else:
            await self._client.set(key, data)
        logger.debug("Saved task", extra={"task_id": task.id})

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        """Retrieve a task from Redis. Returns None if expired or missing.

        Uses GETEX to atomically fetch and refresh TTL in a single round-trip.
        """
        key = self._key(task_id)
        if self._ttl > 0:
            data = await self._client.getex(key, ex=self._ttl)
        else:
            data = await self._client.get(key)
        if data is None:
            return None
        return Task.model_validate_json(data)

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        """Delete a task from Redis."""
        await self._client.delete(self._key(task_id))
        logger.debug("Deleted task", extra={"task_id": task_id})

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.aclose()

    async def __aenter__(self) -> RedisTaskStore:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

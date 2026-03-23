"""AgentRuntime and AgentRuntimeFactory protocols, plus shared base class."""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from fastharness.core.agent import AgentConfig
from fastharness.core.event import Event


@runtime_checkable
class AgentRuntime(Protocol):
    """Protocol for a single agent session that can execute prompts."""

    async def run(self, prompt: str) -> Any:
        """Execute a prompt and return the final result."""
        ...

    def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Execute a prompt, yielding events as they arrive."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by this runtime."""
        ...


@runtime_checkable
class AgentRuntimeFactory(Protocol):
    """Protocol for a factory that manages AgentRuntime lifecycle."""

    async def get_or_create(self, session_key: str, config: AgentConfig) -> AgentRuntime:
        """Return an existing runtime for session_key or create a new one."""
        ...

    async def remove(self, session_key: str) -> None:
        """Remove and close the runtime for session_key."""
        ...

    async def start_cleanup_task(self) -> None:
        """Start background cleanup of idle runtimes."""
        ...

    async def shutdown(self) -> None:
        """Shut down all runtimes and stop cleanup tasks."""
        ...


# ---------------------------------------------------------------------------
# Shared base for session-based factory implementations
# ---------------------------------------------------------------------------


@dataclass
class SessionEntry:
    """Base session entry with TTL metadata. Subclass to add SDK-specific fields."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_stale(self, ttl_minutes: int) -> bool:
        return datetime.now(UTC) - self.last_accessed > timedelta(minutes=ttl_minutes)

    def touch(self) -> None:
        self.last_accessed = datetime.now(UTC)


class BaseSessionFactory:
    """Shared session pool logic for runtime factories.

    Subclasses implement ``_create_session`` and ``_build_runtime`` to provide
    SDK-specific creation, then inherit locking, TTL cleanup, and shutdown.
    """

    def __init__(self, ttl_minutes: int = 15, *, logger: logging.Logger) -> None:
        if ttl_minutes < 1:
            raise ValueError(f"ttl_minutes must be >= 1, got {ttl_minutes}")
        self._ttl_minutes = ttl_minutes
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._log = logger

    # -- Template methods (override in subclasses) --------------------------

    async def _create_session(self, config: AgentConfig) -> SessionEntry:
        """Create a new SessionEntry for the given config."""
        raise NotImplementedError

    def _build_runtime(self, entry: SessionEntry) -> AgentRuntime:
        """Wrap a session entry in an AgentRuntime."""
        raise NotImplementedError

    # -- Shared implementation ----------------------------------------------

    async def get_or_create(self, session_key: str, config: AgentConfig) -> AgentRuntime:
        async with self._lock:
            if session_key in self._sessions:
                entry = self._sessions[session_key]
                entry.touch()
                self._log.info("Reusing session", extra={"session_key": session_key})
                return self._build_runtime(entry)

            entry = await self._create_session(config)
            self._sessions[session_key] = entry
            self._log.info("Created new session", extra={"session_key": session_key})
            return self._build_runtime(entry)

    async def _close_entry(self, entry: SessionEntry) -> None:
        """Close the runtime associated with a session entry."""
        try:
            runtime = self._build_runtime(entry)
            await runtime.aclose()
        except Exception:
            self._log.exception("Error closing runtime")

    async def remove(self, session_key: str) -> None:
        async with self._lock:
            entry = self._sessions.pop(session_key, None)
            if entry:
                await self._close_entry(entry)
                self._log.info("Removed session", extra={"session_key": session_key})

    async def start_cleanup_task(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._log.info("Started session cleanup task")

    async def shutdown(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self._log.info("Shutting down sessions", extra={"count": len(self._sessions)})
            for entry in self._sessions.values():
                await self._close_entry(entry)
            self._sessions.clear()

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                async with self._lock:
                    stale = [
                        k
                        for k, v in self._sessions.items()
                        if v.is_stale(self._ttl_minutes)
                    ]
                    for key in stale:
                        entry = self._sessions.pop(key)
                        await self._close_entry(entry)
                        self._log.info(
                            "Cleaned up stale session", extra={"session_key": key}
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                self._log.exception("Error in cleanup task")

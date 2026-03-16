"""Pydantic DeepAgents implementation of AgentRuntime and AgentRuntimeFactory."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger

try:
    from pydantic_ai._agent_graph import CallToolsNode
    from pydantic_ai_backend import StateBackend
    from pydantic_deep import DeepAgentDeps, create_deep_agent
except ImportError as e:
    raise ImportError(
        "Pydantic DeepAgents is required for DeepAgentsRuntime. "
        "Install with: pip install fastharness[deepagents]"
    ) from e

logger = get_logger("runtime.deepagents")


@dataclass
class _SessionEntry:
    """Tracks a DeepAgents session with TTL metadata."""

    agent: Any  # pydantic_deep agent
    deps: DeepAgentDeps
    message_history: list[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_stale(self, ttl_minutes: int) -> bool:
        return datetime.now(UTC) - self.last_accessed > timedelta(minutes=ttl_minutes)


def _create_agent(config: AgentConfig) -> Any:
    """Create a pydantic-deep agent from AgentConfig."""
    kwargs: dict[str, Any] = {}
    if config.model:
        kwargs["model"] = config.model
    if config.system_prompt:
        kwargs["instructions"] = config.system_prompt
    return create_deep_agent(**kwargs)


class DeepAgentsRuntime:
    """AgentRuntime backed by a Pydantic DeepAgent."""

    def __init__(self, agent: Any, deps: DeepAgentDeps, message_history: list[Any]) -> None:
        self._agent = agent
        self._deps = deps
        self._message_history = message_history

    async def run(self, prompt: str) -> Any:
        """Execute a prompt and return the output."""
        kwargs: dict[str, Any] = {"deps": self._deps}
        if self._message_history:
            kwargs["message_history"] = self._message_history

        result = await self._agent.run(prompt, **kwargs)

        # Preserve history for multi-turn (mutate in-place so factory's entry stays in sync)
        self._message_history[:] = result.all_messages()

        return result.output

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream events from agent execution."""
        kwargs: dict[str, Any] = {"deps": self._deps}
        if self._message_history:
            kwargs["message_history"] = self._message_history

        final_text: str | None = None

        async with self._agent.iter(prompt, **kwargs) as run:
            async for node in run:
                if isinstance(node, CallToolsNode):
                    # Extract tool calls from the model response
                    if hasattr(node, "model_response") and hasattr(node.model_response, "parts"):
                        for part in node.model_response.parts:
                            if hasattr(part, "tool_name"):
                                yield ToolEvent(
                                    tool_name=part.tool_name,
                                    tool_input=getattr(part, "args", {}),
                                )

            result = run.result
            if result:
                self._message_history[:] = result.all_messages()
                output = result.output
                if isinstance(output, str):
                    final_text = output
                    yield TextEvent(text=output)

        yield DoneEvent(final_text=final_text)

    async def aclose(self) -> None:
        """Release resources."""
        self._message_history.clear()


class DeepAgentsRuntimeFactory:
    """AgentRuntimeFactory that manages Pydantic DeepAgent sessions."""

    def __init__(self, ttl_minutes: int = 15) -> None:
        self._ttl_minutes = ttl_minutes
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def get_or_create(self, session_key: str, config: AgentConfig) -> DeepAgentsRuntime:
        """Return an existing runtime or create a new DeepAgent session."""
        async with self._lock:
            if session_key in self._sessions:
                entry = self._sessions[session_key]
                entry.last_accessed = datetime.now(UTC)
                logger.info("Reusing DeepAgents session", extra={"session_key": session_key})
                return DeepAgentsRuntime(entry.agent, entry.deps, entry.message_history)

            agent = _create_agent(config)
            deps = DeepAgentDeps(backend=StateBackend())

            entry = _SessionEntry(agent=agent, deps=deps)
            self._sessions[session_key] = entry

            logger.info("Created new DeepAgents session", extra={"session_key": session_key})
            return DeepAgentsRuntime(agent, deps, entry.message_history)

    async def remove(self, session_key: str) -> None:
        """Remove a session."""
        async with self._lock:
            entry = self._sessions.pop(session_key, None)
            if entry:
                logger.info("Removed DeepAgents session", extra={"session_key": session_key})

    async def start_cleanup_task(self) -> None:
        """Start background cleanup of idle sessions."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started DeepAgents session cleanup task")

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                async with self._lock:
                    stale = [k for k, v in self._sessions.items() if v.is_stale(self._ttl_minutes)]
                    for key in stale:
                        self._sessions.pop(key)
                        logger.info(
                            "Cleaned up stale DeepAgents session", extra={"session_key": key}
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in DeepAgents cleanup task")

    async def shutdown(self) -> None:
        """Shut down all sessions."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            logger.info("Shutting down DeepAgents sessions", extra={"count": len(self._sessions)})
            self._sessions.clear()

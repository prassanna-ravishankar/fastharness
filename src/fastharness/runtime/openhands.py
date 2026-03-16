"""OpenHands SDK implementation of AgentRuntime and AgentRuntimeFactory."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent
from fastharness.logging import get_logger

try:
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.tool import Tool
except ImportError as e:
    raise ImportError(
        "OpenHands SDK is required for OpenHandsRuntime. "
        "Install with: pip install fastharness[openhands]"
    ) from e

logger = get_logger("runtime.openhands")


def _config_to_agent(config: AgentConfig) -> Agent:
    """Convert AgentConfig to an OpenHands Agent."""
    llm = LLM(model=config.model)

    tools = [Tool(name=t) for t in config.tools] if config.tools else None

    return Agent(llm=llm, tools=tools)


@dataclass
class _SessionEntry:
    """Tracks an OpenHands conversation session with TTL metadata."""

    conversation: Conversation
    agent: Agent
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_stale(self, ttl_minutes: int) -> bool:
        return datetime.now(UTC) - self.last_accessed > timedelta(minutes=ttl_minutes)


class OpenHandsRuntime:
    """AgentRuntime backed by an OpenHands Conversation."""

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation

    async def run(self, prompt: str) -> Any:
        """Send a message and run the conversation to completion."""
        self._conversation.send_message(prompt)

        # Conversation.run() is synchronous — offload to a thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conversation.run)

        # Extract final response from conversation events
        events = getattr(self._conversation, "state", None)
        if events and hasattr(events, "events"):
            for event in reversed(events.events):
                content = getattr(event, "content", None)
                if content and isinstance(content, str):
                    return content

        return ""

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream events from the conversation.

        OpenHands SDK doesn't natively support async streaming in the same way,
        so we run the conversation and yield events after completion.
        """
        self._conversation.send_message(prompt)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conversation.run)

        # Yield events from conversation state
        events = getattr(self._conversation, "state", None)
        final_text: str | None = None

        if events and hasattr(events, "events"):
            for event in events.events:
                content = getattr(event, "content", None)
                if content and isinstance(content, str):
                    final_text = content
                    yield TextEvent(text=content)

        yield DoneEvent(final_text=final_text)

    async def aclose(self) -> None:
        """Release conversation resources."""
        # OpenHands Conversation doesn't have an explicit close
        pass


class OpenHandsRuntimeFactory:
    """AgentRuntimeFactory that manages OpenHands conversation sessions."""

    def __init__(self, ttl_minutes: int = 15, workspace: str | None = None) -> None:
        self._ttl_minutes = ttl_minutes
        self._workspace = workspace
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def get_or_create(self, session_key: str, config: AgentConfig) -> OpenHandsRuntime:
        """Return an existing runtime or create a new OpenHands conversation."""
        async with self._lock:
            if session_key in self._sessions:
                entry = self._sessions[session_key]
                entry.last_accessed = datetime.now(UTC)
                logger.info("Reusing OpenHands session", extra={"session_key": session_key})
                return OpenHandsRuntime(entry.conversation)

            agent = _config_to_agent(config)

            kwargs: dict[str, Any] = {"agent": agent}
            if self._workspace:
                kwargs["workspace"] = self._workspace
            if config.max_turns:
                kwargs["max_iteration_per_run"] = config.max_turns

            conversation = Conversation(**kwargs)

            entry = _SessionEntry(conversation=conversation, agent=agent)
            self._sessions[session_key] = entry

            logger.info("Created new OpenHands session", extra={"session_key": session_key})
            return OpenHandsRuntime(conversation)

    async def remove(self, session_key: str) -> None:
        """Remove a session."""
        async with self._lock:
            entry = self._sessions.pop(session_key, None)
            if entry:
                logger.info("Removed OpenHands session", extra={"session_key": session_key})

    async def start_cleanup_task(self) -> None:
        """Start background cleanup of idle sessions."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started OpenHands session cleanup task")

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                async with self._lock:
                    stale = [k for k, v in self._sessions.items() if v.is_stale(self._ttl_minutes)]
                    for key in stale:
                        self._sessions.pop(key)
                        logger.info(
                            "Cleaned up stale OpenHands session",
                            extra={"session_key": key},
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in OpenHands cleanup task")

    async def shutdown(self) -> None:
        """Shut down all sessions."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            logger.info("Shutting down OpenHands sessions", extra={"count": len(self._sessions)})
            self._sessions.clear()

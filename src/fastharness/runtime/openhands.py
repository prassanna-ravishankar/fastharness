"""OpenHands SDK implementation of AgentRuntime and AgentRuntimeFactory."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent
from fastharness.logging import get_logger
from fastharness.runtime.base import BaseSessionFactory, SessionEntry

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
    tools = [Tool(name=t) for t in config.tools] if config.tools else []
    return Agent(llm=llm, tools=tools)


@dataclass
class _OHSession(SessionEntry):
    """OpenHands-specific session entry."""

    conversation: Any = field(default=None)  # Conversation
    agent: Any = field(default=None)  # Agent


class OpenHandsRuntime:
    """AgentRuntime backed by an OpenHands Conversation."""

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation

    @staticmethod
    def _extract_agent_text(events: list[Any]) -> str:
        """Extract the last agent message text from conversation events."""
        for event in reversed(events):
            if getattr(event, "source", None) != "agent":
                continue
            llm_msg = getattr(event, "llm_message", None)
            if llm_msg is None:
                continue
            for block in getattr(llm_msg, "content", []):
                text = getattr(block, "text", None)
                if text:
                    return text
        return ""

    async def run(self, prompt: str) -> Any:
        """Send a message and run the conversation to completion."""
        self._conversation.send_message(prompt)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conversation.run)

        state = getattr(self._conversation, "state", None)
        events = getattr(state, "events", [])
        return self._extract_agent_text(events)

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream events from the conversation.

        OpenHands SDK runs synchronously, so we yield events after completion.
        """
        self._conversation.send_message(prompt)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conversation.run)

        state = getattr(self._conversation, "state", None)
        raw_events = getattr(state, "events", [])
        final_text: str | None = None

        for event in raw_events:
            if getattr(event, "source", None) != "agent":
                continue
            llm_msg = getattr(event, "llm_message", None)
            if llm_msg is None:
                continue
            for block in getattr(llm_msg, "content", []):
                text = getattr(block, "text", None)
                if text:
                    final_text = text
                    yield TextEvent(text=text)

        yield DoneEvent(final_text=final_text)

    async def aclose(self) -> None:
        """Release conversation resources."""
        pass


class OpenHandsRuntimeFactory(BaseSessionFactory):
    """AgentRuntimeFactory that manages OpenHands conversation sessions."""

    def __init__(self, ttl_minutes: int = 15, workspace: str | None = None) -> None:
        super().__init__(ttl_minutes=ttl_minutes, logger=logger)
        self._workspace = workspace

    async def _create_session(self, config: AgentConfig, session_key: str = "") -> _OHSession:
        agent = _config_to_agent(config)

        kwargs: dict[str, Any] = {"agent": agent}
        if self._workspace:
            kwargs["workspace"] = self._workspace
        if config.max_turns:
            kwargs["max_iteration_per_run"] = config.max_turns

        conversation = Conversation(**kwargs)
        return _OHSession(conversation=conversation, agent=agent)

    def _build_runtime(self, entry: SessionEntry) -> OpenHandsRuntime:
        if not isinstance(entry, _OHSession):
            raise TypeError(f"Expected _OHSession, got {type(entry).__name__}")
        return OpenHandsRuntime(entry.conversation)

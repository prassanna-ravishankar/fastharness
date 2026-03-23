"""OpenClaw bridge — expose any OpenClaw agent as an A2A service.

Usage::

    from fastharness.bridges.openclaw import OpenClawBridge

    bridge = OpenClawBridge("ws://localhost:18789")

    # Expose a single agent
    app = bridge.expose("research-bot", description="Research assistant")

    # Or expose multiple agents on one service
    harness = bridge.to_harness("my-service")
    bridge.add_agent(harness, "research-bot", description="Research")
    bridge.add_agent(harness, "coder-bot", description="Coding assistant")
    app = harness.app

Requires: pip install fastharness[openclaw]
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.core.skill import Skill
from fastharness.logging import get_logger
from fastharness.runtime.base import BaseSessionFactory, SessionEntry

if TYPE_CHECKING:
    from fastapi import FastAPI

    from fastharness.app import FastHarness

try:
    from openclaw_sdk import OpenClawClient  # ty: ignore[unresolved-import]
except ImportError as e:
    raise ImportError(
        "OpenClaw SDK is required for the OpenClaw bridge. "
        "Install with: pip install fastharness[openclaw]"
    ) from e

logger = get_logger("bridges.openclaw")


@dataclass
class _OpenClawSession(SessionEntry):
    """Session entry holding an OpenClaw client and conversation."""

    client: Any = field(default=None)  # OpenClawClient
    agent: Any = field(default=None)  # openclaw_sdk Agent
    conversation: Any = field(default=None)  # openclaw_sdk Conversation


class OpenClawRuntime:
    """AgentRuntime that proxies to an OpenClaw agent via the SDK."""

    def __init__(self, agent: Any, conversation: Any) -> None:
        self._agent = agent
        self._conversation = conversation

    async def run(self, prompt: str) -> Any:
        """Send a message via the conversation (multi-turn with server-side history)."""
        result = await self._conversation.say(prompt)
        return result.content

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream typed events from the OpenClaw agent."""
        stream = await self._agent.execute_stream_typed(prompt)
        final_text: str | None = None
        metrics: dict[str, Any] = {}

        async for event in stream:
            event_cls = type(event).__name__

            if event_cls == "ContentEvent":
                text = event.text
                final_text = text
                yield TextEvent(text=text)

            elif event_cls == "ToolCallEvent":
                yield ToolEvent(
                    tool_name=getattr(event, "tool", ""),
                    tool_input=getattr(event, "input", {}),
                )

            elif event_cls == "DoneEvent":
                if hasattr(event, "content") and event.content:
                    final_text = event.content
                token_usage = getattr(event, "token_usage", None)
                if token_usage:
                    metrics = {
                        "input_tokens": getattr(token_usage, "input_tokens", None),
                        "output_tokens": getattr(token_usage, "output_tokens", None),
                    }
                yield DoneEvent(final_text=final_text, metrics=metrics)
                return

            elif event_cls == "ErrorEvent":
                msg = getattr(event, "message", "OpenClaw agent error")
                raise RuntimeError(f"OpenClaw agent error: {msg}")

        # Stream ended without DoneEvent
        yield DoneEvent(final_text=final_text, metrics=metrics)

    async def aclose(self) -> None:
        """Nothing to close — client lifecycle managed by factory."""
        pass


class OpenClawRuntimeFactory(BaseSessionFactory):
    """Factory that manages OpenClaw client connections per session."""

    def __init__(
        self,
        gateway_url: str | None = None,
        ttl_minutes: int = 15,
    ) -> None:
        super().__init__(ttl_minutes=ttl_minutes, logger=logger)
        self._gateway_url = gateway_url

    async def _create_session(self, config: AgentConfig) -> _OpenClawSession:
        connect_kwargs: dict[str, Any] = {}
        if self._gateway_url:
            connect_kwargs["gateway_ws_url"] = self._gateway_url

        client = await OpenClawClient.connect(**connect_kwargs)
        agent = client.get_agent(config.name)
        conversation = agent.conversation(f"fastharness:{config.name}")

        logger.info(
            "Connected to OpenClaw agent",
            extra={"agent_id": config.name, "gateway": self._gateway_url or "auto"},
        )
        return _OpenClawSession(client=client, agent=agent, conversation=conversation)

    def _build_runtime(self, entry: SessionEntry) -> OpenClawRuntime:
        assert isinstance(entry, _OpenClawSession)
        return OpenClawRuntime(entry.agent, entry.conversation)

    async def _close_entry(self, entry: SessionEntry) -> None:
        """Close the WebSocket client when a session is evicted."""
        if isinstance(entry, _OpenClawSession) and entry.client is not None:
            try:
                await entry.client.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing OpenClaw client")


class OpenClawBridge:
    """Bridge adapter that wraps OpenClaw agents as A2A endpoints.

    The simplest way to make OpenClaw agents A2A-accessible::

        bridge = OpenClawBridge("ws://localhost:18789")
        app = bridge.expose("my-agent", description="My OpenClaw agent")
        # Run with: uvicorn mymodule:app

    For multiple agents on one service::

        harness = bridge.to_harness("multi-agent-service")
        bridge.add_agent(harness, "agent-1", description="First agent")
        bridge.add_agent(harness, "agent-2", description="Second agent")
        app = harness.app
    """

    def __init__(
        self,
        gateway_url: str | None = None,
        ttl_minutes: int = 15,
    ) -> None:
        """Initialize the bridge.

        Args:
            gateway_url: WebSocket URL of the OpenClaw gateway.
                Auto-detected from OPENCLAW_GATEWAY_WS_URL env var if not set.
            ttl_minutes: Session TTL for idle connections.
        """
        self._gateway_url = gateway_url
        self._ttl_minutes = ttl_minutes

    def _make_factory(self) -> OpenClawRuntimeFactory:
        return OpenClawRuntimeFactory(
            gateway_url=self._gateway_url,
            ttl_minutes=self._ttl_minutes,
        )

    def to_harness(
        self,
        name: str = "openclaw-bridge",
        description: str = "OpenClaw agents exposed via A2A",
        **kwargs: Any,
    ) -> FastHarness:
        """Create a FastHarness instance wired to this OpenClaw gateway.

        Use with ``add_agent()`` to expose multiple agents on one service.
        """
        from fastharness import FastHarness

        return FastHarness(
            name=name,
            description=description,
            runtime_factory=self._make_factory(),
            **kwargs,
        )

    def add_agent(
        self,
        harness: FastHarness,
        agent_id: str,
        description: str = "",
        skills: list[Skill] | None = None,
    ) -> None:
        """Register an OpenClaw agent on an existing harness.

        Args:
            harness: The FastHarness instance (from ``to_harness()``).
            agent_id: The OpenClaw agent ID on the gateway.
            description: Human-readable description for the A2A agent card.
            skills: A2A skills to advertise. Defaults to a generic skill.
        """
        if not skills:
            skills = [
                Skill(
                    id=agent_id,
                    name=agent_id,
                    description=description or f"OpenClaw agent: {agent_id}",
                )
            ]

        harness.agent(
            name=agent_id,
            description=description or f"OpenClaw agent: {agent_id}",
            skills=skills,
        )

    def expose(
        self,
        agent_id: str,
        description: str = "",
        skills: list[Skill] | None = None,
        **kwargs: Any,
    ) -> FastAPI:
        """One-liner: expose a single OpenClaw agent as an A2A endpoint.

        Args:
            agent_id: The OpenClaw agent ID on the gateway.
            description: Human-readable description.
            skills: A2A skills to advertise.
            **kwargs: Extra args passed to FastHarness constructor.

        Returns:
            A FastAPI app ready for uvicorn.
        """
        harness = self.to_harness(
            name=agent_id,
            description=description or f"OpenClaw agent: {agent_id}",
            **kwargs,
        )
        self.add_agent(harness, agent_id, description=description, skills=skills)
        return harness.app

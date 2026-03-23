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

from typing import TYPE_CHECKING, Any

from fastharness.core.skill import Skill
from fastharness.runtime.openclaw import OpenClawRuntimeFactory

if TYPE_CHECKING:
    from fastapi import FastAPI

    from fastharness.app import FastHarness


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
        desc = description or f"OpenClaw agent: {agent_id}"
        if not skills:
            skills = [Skill(id=agent_id, name=agent_id, description=desc)]

        harness.agent(name=agent_id, description=desc, skills=skills)

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
        desc = description or f"OpenClaw agent: {agent_id}"
        harness = self.to_harness(name=agent_id, description=desc, **kwargs)
        self.add_agent(harness, agent_id, description=description, skills=skills)
        return harness.app

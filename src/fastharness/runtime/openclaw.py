"""OpenClaw runtime — AgentRuntime/AgentRuntimeFactory for OpenClaw agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger
from fastharness.runtime.base import BaseSessionFactory, SessionEntry

try:
    from openclaw_sdk import OpenClawClient  # ty: ignore[unresolved-import]
except ImportError as e:
    raise ImportError(
        "OpenClaw SDK is required. Install with: pip install fastharness[openclaw]"
    ) from e

logger = get_logger("runtime.openclaw")


@dataclass
class _OpenClawSession(SessionEntry):
    """Session entry holding an OpenClaw client and conversation."""

    client: Any = field(default=None)
    agent: Any = field(default=None)
    conversation: Any = field(default=None)


class OpenClawRuntime:
    """AgentRuntime that proxies to an OpenClaw agent via the SDK."""

    def __init__(self, agent: Any, conversation: Any, client: Any) -> None:
        self._agent = agent
        self._conversation = conversation
        self._client = client

    async def run(self, prompt: str) -> Any:
        """Send a message via the conversation (multi-turn with server-side history)."""
        result = await self._conversation.say(prompt)
        return result.content

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream typed events, using the conversation for multi-turn history."""
        stream = await self._conversation.stream(prompt)
        chunks: list[str] = []
        metrics: dict[str, Any] = {}

        async for event in stream:
            event_cls = type(event).__name__

            if event_cls == "ContentEvent":
                chunks.append(event.text)
                yield TextEvent(text=event.text)

            elif event_cls == "ToolCallEvent":
                yield ToolEvent(
                    tool_name=getattr(event, "tool", ""),
                    tool_input=getattr(event, "input", {}),
                )

            elif event_cls == "DoneEvent":
                final = getattr(event, "content", None) or "".join(chunks)
                token_usage = getattr(event, "token_usage", None)
                if token_usage:
                    metrics = {
                        "input_tokens": getattr(token_usage, "input_tokens", None),
                        "output_tokens": getattr(token_usage, "output_tokens", None),
                    }
                yield DoneEvent(final_text=final, metrics=metrics)
                return

            elif event_cls == "ErrorEvent":
                msg = getattr(event, "message", "OpenClaw agent error")
                raise RuntimeError(f"OpenClaw agent error: {msg}")

        yield DoneEvent(final_text="".join(chunks), metrics=metrics)

    async def aclose(self) -> None:
        """Close the underlying WebSocket client."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing OpenClaw client")


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
        return OpenClawRuntime(entry.agent, entry.conversation, entry.client)

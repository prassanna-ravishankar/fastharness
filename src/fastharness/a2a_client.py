"""Convenience client for talking to FastHarness A2A agents.

Wraps the A2A SDK client with a simpler API focused on sending text messages
and getting text responses.
"""

import uuid
from typing import Any

import httpx
from a2a.client import A2AClient
from a2a.types import Message as A2AMessage
from a2a.types import (
    MessageSendParams,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

from fastharness.logging import get_logger
from fastharness.worker.converter import MessageConverter, _text_part

logger = get_logger("a2a_client")


class FastHarnessClient:
    """Simple client for communicating with a FastHarness agent.

    Usage::

        async with FastHarnessClient("http://localhost:8000") as client:
            reply = await client.send("Hello!")
            print(reply)

            # Multi-turn with same context
            reply2 = await client.send("What did I just say?", context_id="conv-1")
    """

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._httpx: httpx.AsyncClient | None = None
        self._a2a: A2AClient | None = None
        self._msg_counter = 0

    async def __aenter__(self) -> "FastHarnessClient":
        self._httpx = httpx.AsyncClient(timeout=self._timeout)
        self._a2a = A2AClient(httpx_client=self._httpx, url=self._url)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._httpx:
            await self._httpx.aclose()

    def _next_msg_id(self) -> str:
        self._msg_counter += 1
        return f"msg-{self._msg_counter}"

    def _build_message(
        self,
        text: str,
        context_id: str | None = None,
        skill_id: str | None = None,
    ) -> tuple[A2AMessage, dict[str, Any] | None]:
        """Build an A2A Message and optional metadata."""
        msg = A2AMessage(
            role=Role.user,
            parts=[_text_part(text)],
            message_id=self._next_msg_id(),
            context_id=context_id or str(uuid.uuid4()),
        )
        metadata = {"skill_id": skill_id} if skill_id else None
        return msg, metadata

    async def send(
        self,
        text: str,
        *,
        context_id: str | None = None,
        skill_id: str | None = None,
    ) -> str:
        """Send a text message and return the agent's text response.

        Args:
            text: The message to send.
            context_id: Conversation context ID for multi-turn. Auto-generated if omitted.
            skill_id: Target a specific agent skill.

        Returns:
            The agent's text response.
        """
        assert self._a2a is not None, "Use 'async with' to initialize the client"
        msg, metadata = self._build_message(text, context_id, skill_id)

        request = SendMessageRequest(
            id=self._msg_counter,
            params=MessageSendParams(message=msg, metadata=metadata),
        )
        response = await self._a2a.send_message(request)

        # Unwrap union: SendMessageResponse.root is Success | Error
        inner = response.root
        if hasattr(inner, "error"):
            raise RuntimeError(f"A2A error: {inner.error}")
        result = inner.result

        # Result is Message | Task
        return _extract_response_text(result)

    async def stream(
        self,
        text: str,
        *,
        context_id: str | None = None,
        skill_id: str | None = None,
    ):
        """Send a message and yield text chunks as they arrive.

        Args:
            text: The message to send.
            context_id: Conversation context ID for multi-turn.
            skill_id: Target a specific agent skill.

        Yields:
            Text chunks as they arrive from the agent.
        """
        assert self._a2a is not None, "Use 'async with' to initialize the client"
        msg, metadata = self._build_message(text, context_id, skill_id)

        request = SendStreamingMessageRequest(
            id=self._msg_counter,
            params=MessageSendParams(message=msg, metadata=metadata),
        )
        async for response in self._a2a.send_message_streaming(request):
            inner = response.root
            if hasattr(inner, "error"):
                continue
            event = inner.result
            if event is None:
                continue
            # Artifact update — extract text chunks
            artifact = getattr(event, "artifact", None)
            if artifact is not None:
                chunk = MessageConverter.extract_text_from_parts(artifact.parts)
                if chunk:
                    yield chunk

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the agent card."""
        assert self._a2a is not None, "Use 'async with' to initialize the client"
        card = await self._a2a.get_card()
        return card.model_dump()


def _extract_response_text(result: Any) -> str:
    """Extract text from a Message or Task response."""
    if result is None:
        return ""

    # Message response — has parts directly
    if hasattr(result, "parts") and result.parts:
        return MessageConverter.extract_text_from_parts(result.parts)

    # Task response — check artifacts first, then history
    if hasattr(result, "artifacts") and result.artifacts:
        for artifact in result.artifacts:
            text_out = MessageConverter.extract_text_from_parts(artifact.parts)
            if text_out:
                return text_out

    if hasattr(result, "history") and result.history:
        for hist_msg in reversed(result.history):
            if hist_msg.role == Role.agent:
                return MessageConverter.extract_text_from_parts(hist_msg.parts)

    return ""



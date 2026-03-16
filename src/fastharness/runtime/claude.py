"""Claude SDK implementation of AgentRuntime and AgentRuntimeFactory."""

from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger
from fastharness.worker.client_pool import ClientPool

logger = get_logger("runtime.claude")


def _config_to_options(config: AgentConfig) -> ClaudeAgentOptions:
    """Convert AgentConfig to ClaudeAgentOptions."""
    return ClaudeAgentOptions(
        system_prompt=config.system_prompt,
        allowed_tools=config.tools,
        model=config.model,
        max_turns=config.max_turns,
        mcp_servers=config.mcp_servers,
        setting_sources=cast(
            list[Literal["user", "project", "local"]] | None, config.setting_sources
        ),
        output_format=config.output_format,
        permission_mode="bypassPermissions",
    )


class ClaudeRuntime:
    """AgentRuntime backed by a pooled ClaudeSDKClient."""

    def __init__(self, client: ClaudeSDKClient) -> None:
        self._client = client

    async def run(self, prompt: str) -> Any:
        """Execute a prompt and return the final text or structured output."""
        final_text = ""
        structured_output: Any = None

        await self._client.query(prompt)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text
            elif isinstance(message, ResultMessage):
                if message.result:
                    final_text = message.result
                structured_output = getattr(message, "structured_output", None)
                break

        if structured_output is not None:
            return structured_output
        return final_text

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Execute a prompt, yielding events as they arrive."""
        final_text: str | None = None

        await self._client.query(prompt)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text
                        yield TextEvent(text=block.text)
                    elif hasattr(block, "name") and hasattr(block, "input"):
                        yield ToolEvent(
                            tool_name=getattr(block, "name", ""),
                            tool_input=getattr(block, "input", {}),
                        )
            elif isinstance(message, ResultMessage):
                if message.result:
                    final_text = message.result
                usage = getattr(message, "usage", None)
                usage_dict = usage if isinstance(usage, dict) else {}
                yield DoneEvent(
                    final_text=final_text,
                    structured_output=getattr(message, "structured_output", None),
                    metrics={
                        "total_cost_usd": getattr(message, "total_cost_usd", None),
                        "input_tokens": usage_dict.get("input_tokens"),
                        "output_tokens": usage_dict.get("output_tokens"),
                        "duration_ms": getattr(message, "duration_ms", 0),
                        "num_turns": getattr(message, "num_turns", 1),
                        "session_id": getattr(message, "session_id", "unknown"),
                    },
                )
                break

    async def aclose(self) -> None:
        """Close the underlying SDK client."""
        await self._client.__aexit__(None, None, None)


class ClaudeRuntimeFactory:
    """AgentRuntimeFactory backed by ClientPool."""

    def __init__(self, ttl_minutes: int = 15) -> None:
        self._pool = ClientPool(ttl_minutes=ttl_minutes)

    async def get_or_create(self, session_key: str, config: AgentConfig) -> ClaudeRuntime:
        """Return a ClaudeRuntime for session_key, creating one if needed."""
        options = _config_to_options(config)
        client, is_new = await self._pool.get_or_create(session_key, options)
        logger.info(
            "Retrieved runtime from pool",
            extra={"session_key": session_key, "is_new": is_new},
        )
        return ClaudeRuntime(client)

    async def remove(self, session_key: str) -> None:
        """Remove and close the runtime for session_key."""
        await self._pool.remove(session_key)

    async def start_cleanup_task(self) -> None:
        """Start background TTL cleanup."""
        await self._pool.start_cleanup_task()

    async def shutdown(self) -> None:
        """Shut down the pool and close all runtimes."""
        await self._pool.shutdown()

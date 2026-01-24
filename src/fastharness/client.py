"""HarnessClient - Simplified wrapper over Claude SDK for agent execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    PermissionMode,
    ResultMessage,
    TextBlock,
)

from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger

logger = get_logger("client")

# Permission mode type alias
PermissionModeType = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


@dataclass
class HarnessClient:
    """Thin wrapper over ClaudeSDKClient with simplified API.

    Provides a minimal interface for executing Claude agents within
    an A2A task context.
    """

    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    max_turns: int | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    permission_mode: PermissionModeType = "bypassPermissions"

    def _build_options(self, **overrides: Any) -> ClaudeCodeOptions:
        """Build ClaudeCodeOptions from client config and overrides."""
        permission_mode: PermissionMode = overrides.get(
            "permission_mode", self.permission_mode
        )
        opts = ClaudeCodeOptions(
            system_prompt=overrides.get("system_prompt", self.system_prompt),
            allowed_tools=overrides.get("tools", self.tools),
            model=overrides.get("model", self.model),
            max_turns=overrides.get("max_turns", self.max_turns),
            mcp_servers=overrides.get("mcp_servers", self.mcp_servers),
            cwd=overrides.get("cwd", self.cwd),
            permission_mode=permission_mode,
        )
        return opts

    async def run(self, prompt: str, **opts: Any) -> str:
        """Execute full agent loop, return final text.

        Args:
            prompt: The user prompt to send to the agent.
            **opts: Override options (system_prompt, tools, model, max_turns, etc.)

        Returns:
            The final text response from the agent.

        Raises:
            RuntimeError: If Claude SDK execution fails.
        """
        options = self._build_options(**opts)
        final_text = ""

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text = block.text
                    elif isinstance(message, ResultMessage):
                        if message.result:
                            final_text = message.result
                        break
        except Exception as e:
            logger.exception(
                "Claude SDK execution failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(
                f"Agent execution failed: {type(e).__name__}: {e}"
            ) from e

        return final_text

    async def stream(self, prompt: str, **opts: Any) -> AsyncIterator[Event]:
        """Execute with streaming, yield events.

        Args:
            prompt: The user prompt to send to the agent.
            **opts: Override options (system_prompt, tools, model, max_turns, etc.)

        Yields:
            Event objects (TextEvent, ToolEvent, DoneEvent) as execution progresses.

        Raises:
            RuntimeError: If Claude SDK execution fails.
        """
        options = self._build_options(**opts)
        final_text: str | None = None

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text = block.text
                                yield TextEvent(text=block.text)
                            elif hasattr(block, "name") and hasattr(block, "input"):
                                # ToolUseBlock
                                yield ToolEvent(
                                    tool_name=block.name,
                                    tool_input=block.input,
                                )
                    elif isinstance(message, ResultMessage):
                        if message.result:
                            final_text = message.result
                        yield DoneEvent(final_text=final_text)
                        break
        except Exception as e:
            logger.exception(
                "Claude SDK streaming failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(
                f"Agent streaming failed: {type(e).__name__}: {e}"
            ) from e

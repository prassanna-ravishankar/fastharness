"""HarnessClient - Simplified wrapper over Claude SDK for agent execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger
from fastharness.tracing import chat_span, record_usage, tool_span

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
    setting_sources: list[str] | None = field(default_factory=lambda: ["project"])
    output_format: dict[str, Any] | None = None
    tracing: bool = False

    def _build_options(self, **overrides: Any) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions, merging client config with overrides.

        FastHarness field names (tools) map to SDK names (allowed_tools).
        """
        # Map FastHarness names to SDK names
        sdk_overrides: dict[str, Any] = {}
        if "tools" in overrides:
            sdk_overrides["allowed_tools"] = overrides.pop("tools")

        # Merge defaults from client config with overrides
        opts_dict: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "allowed_tools": self.tools,
            "model": self.model,
            "max_turns": self.max_turns,
            "mcp_servers": self.mcp_servers,
            "cwd": self.cwd,
            "permission_mode": self.permission_mode,
            "setting_sources": self.setting_sources,
            "output_format": self.output_format,
        }
        opts_dict.update(sdk_overrides)
        opts_dict.update(overrides)

        return ClaudeAgentOptions(**opts_dict)

    async def run(self, prompt: str, **opts: Any) -> Any:
        """Execute full agent loop, return final text or structured output.

        When output_format is configured, returns the structured output
        (parsed from the agent's JSON schema response). Otherwise returns
        the final text response as a string.

        Args:
            prompt: The user prompt to send to the agent.
            **opts: Override options (system_prompt, tools, model, max_turns,
                output_format, etc.)

        Returns:
            Structured output if output_format is set and the agent returned
            structured data, otherwise the final text response string.

        Raises:
            RuntimeError: If Claude Agent SDK execution fails.
        """
        options = self._build_options(**opts)
        final_text = ""
        structured_output: Any = None

        try:
            with chat_span(self.model) if self.tracing else _nullcontext() as span:
                async with ClaudeSDKClient(options) as client:
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    final_text = block.text
                                elif (
                                    self.tracing
                                    and hasattr(block, "name")
                                    and hasattr(block, "input")
                                ):
                                    with tool_span(
                                        getattr(block, "name", ""),
                                        getattr(block, "id", ""),
                                    ):
                                        pass
                        elif isinstance(message, ResultMessage):
                            if message.result:
                                final_text = message.result
                            structured_output = getattr(message, "structured_output", None)
                            if self.tracing:
                                usage = getattr(message, "usage", None)
                                usage_dict = usage if isinstance(usage, dict) else None
                                record_usage(span, usage_dict)
                            break
        except Exception as e:
            logger.exception(
                "Claude Agent SDK execution failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(f"Agent execution failed: {type(e).__name__}: {e}") from e

        if structured_output is not None:
            return structured_output

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
            with chat_span(self.model) if self.tracing else _nullcontext() as span:
                async with ClaudeSDKClient(options) as client:
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    final_text = block.text
                                    yield TextEvent(text=block.text)
                                elif hasattr(block, "name") and hasattr(block, "input"):
                                    tool_name = getattr(block, "name", "")
                                    tool_input = getattr(block, "input", {})
                                    if self.tracing:
                                        with tool_span(
                                            tool_name,
                                            getattr(block, "id", ""),
                                        ):
                                            pass
                                    yield ToolEvent(
                                        tool_name=tool_name,
                                        tool_input=tool_input,
                                    )
                        elif isinstance(message, ResultMessage):
                            if message.result:
                                final_text = message.result
                            if self.tracing:
                                usage = getattr(message, "usage", None)
                                usage_dict = usage if isinstance(usage, dict) else None
                                record_usage(span, usage_dict)
                            yield DoneEvent(
                                final_text=final_text,
                                structured_output=getattr(message, "structured_output", None),
                            )
                            break
        except Exception as e:
            logger.exception(
                "Claude Agent SDK streaming failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(f"Agent streaming failed: {type(e).__name__}: {e}") from e


class _nullcontext:
    """Minimal context manager that yields None (avoids contextlib import)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc: Any):
        return False

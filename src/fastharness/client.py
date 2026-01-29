"""HarnessClient - Simplified wrapper over Claude SDK for agent execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger

if TYPE_CHECKING:
    from fastharness.step_logger import StepLogger
    from fastharness.telemetry import TelemetryCallback

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
    telemetry_callbacks: list["TelemetryCallback"] = field(default_factory=list)
    step_logger: "StepLogger | None" = None
    enable_step_logging: bool = False

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
        }
        opts_dict.update(sdk_overrides)
        opts_dict.update(overrides)

        return ClaudeAgentOptions(**opts_dict)

    async def _emit_telemetry(self, result_msg: ResultMessage, task_id: str = "unknown") -> None:
        """Extract metrics from ResultMessage and notify callbacks."""
        if not self.telemetry_callbacks:
            return

        from fastharness.telemetry import ExecutionMetrics

        usage = getattr(result_msg, "usage", None)
        metrics = ExecutionMetrics(
            task_id=task_id,
            session_id=getattr(result_msg, "session_id", "unknown"),
            total_cost_usd=getattr(result_msg, "total_cost_usd", None),
            input_tokens=(usage.get("input_tokens") if isinstance(usage, dict) else None),
            output_tokens=(usage.get("output_tokens") if isinstance(usage, dict) else None),
            cache_read_tokens=(
                usage.get("cache_read_input_tokens") if isinstance(usage, dict) else None
            ),
            cache_write_tokens=(
                usage.get("cache_creation_input_tokens") if isinstance(usage, dict) else None
            ),
            duration_ms=getattr(result_msg, "duration_ms", 0),
            duration_api_ms=getattr(result_msg, "duration_api_ms", 0),
            num_turns=getattr(result_msg, "num_turns", 1),
            status="success" if not getattr(result_msg, "is_error", False) else "error",
            timestamp=datetime.now(UTC),
        )

        for callback in self.telemetry_callbacks:
            await callback.on_complete(metrics)

    async def _log_step(self, event: dict[str, Any]) -> None:
        """Log a step if logging is enabled."""
        if self.enable_step_logging and self.step_logger:
            from fastharness.step_logger import StepEvent

            step_event = StepEvent(
                step_type=event["step_type"],
                turn_number=event["turn_number"],
                data=event["data"],
            )
            await self.step_logger.log_step(step_event)

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
        turn_number = 0

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text = block.text
                                await self._log_step(
                                    {
                                        "step_type": "assistant_message",
                                        "turn_number": turn_number,
                                        "data": {"text": block.text},
                                    }
                                )
                            elif hasattr(block, "name") and hasattr(block, "input"):
                                await self._log_step(
                                    {
                                        "step_type": "tool_call",
                                        "turn_number": turn_number,
                                        "data": {
                                            "name": getattr(block, "name", ""),
                                            "id": getattr(block, "id", "unknown"),
                                            "input": getattr(block, "input", {}),
                                        },
                                    }
                                )
                    elif isinstance(message, ResultMessage):
                        await self._log_step(
                            {
                                "step_type": "turn_complete",
                                "turn_number": turn_number,
                                "data": {
                                    "cost_usd": getattr(message, "total_cost_usd", None),
                                    "usage": getattr(message, "usage", None),
                                },
                            }
                        )
                        await self._emit_telemetry(message)
                        if message.result:
                            final_text = message.result
                        turn_number += 1
                        break
        except Exception as e:
            logger.exception(
                "Claude SDK execution failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(f"Agent execution failed: {type(e).__name__}: {e}") from e

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
        turn_number = 0

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text = block.text
                                await self._log_step(
                                    {
                                        "step_type": "assistant_message",
                                        "turn_number": turn_number,
                                        "data": {"text": block.text},
                                    }
                                )
                                yield TextEvent(text=block.text)
                            elif hasattr(block, "name") and hasattr(block, "input"):
                                # ToolUseBlock
                                tool_name = getattr(block, "name", "")
                                tool_input = getattr(block, "input", {})
                                await self._log_step(
                                    {
                                        "step_type": "tool_call",
                                        "turn_number": turn_number,
                                        "data": {
                                            "name": tool_name,
                                            "id": getattr(block, "id", "unknown"),
                                            "input": tool_input,
                                        },
                                    }
                                )
                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                )
                    elif isinstance(message, ResultMessage):
                        await self._log_step(
                            {
                                "step_type": "turn_complete",
                                "turn_number": turn_number,
                                "data": {
                                    "cost_usd": getattr(message, "total_cost_usd", None),
                                    "usage": getattr(message, "usage", None),
                                },
                            }
                        )
                        await self._emit_telemetry(message)
                        if message.result:
                            final_text = message.result
                        yield DoneEvent(final_text=final_text)
                        turn_number += 1
                        break
        except Exception as e:
            logger.exception(
                "Claude SDK streaming failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(f"Agent streaming failed: {type(e).__name__}: {e}") from e

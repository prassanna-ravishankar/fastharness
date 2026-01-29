"""HarnessClient - Simplified wrapper over Claude SDK for agent execution."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from fastharness.step_logger import StepEvent, StepLogger, StepType
from fastharness.telemetry import ExecutionMetrics, TelemetryCallback

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
    telemetry_callbacks: list[TelemetryCallback] = field(default_factory=list)
    step_logger: StepLogger | None = None

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

    async def _emit_telemetry(self, result_msg: ResultMessage, task_id: str = "unknown") -> None:
        """Extract metrics from ResultMessage and notify callbacks."""
        if not self.telemetry_callbacks:
            return

        usage = getattr(result_msg, "usage", None)
        usage_dict = usage if isinstance(usage, dict) else {}
        metrics = ExecutionMetrics(
            task_id=task_id,
            session_id=getattr(result_msg, "session_id", "unknown"),
            total_cost_usd=getattr(result_msg, "total_cost_usd", None),
            input_tokens=usage_dict.get("input_tokens"),
            output_tokens=usage_dict.get("output_tokens"),
            cache_read_tokens=usage_dict.get("cache_read_input_tokens"),
            cache_write_tokens=usage_dict.get("cache_creation_input_tokens"),
            duration_ms=getattr(result_msg, "duration_ms", 0),
            duration_api_ms=getattr(result_msg, "duration_api_ms", 0),
            num_turns=getattr(result_msg, "num_turns", 1),
            status="error" if getattr(result_msg, "is_error", False) else "success",
            timestamp=datetime.now(UTC),
        )

        for callback in self.telemetry_callbacks:
            try:
                await callback.on_complete(metrics)
            except Exception:
                logger.exception(
                    "Telemetry callback failed",
                    extra={"callback": type(callback).__name__, "task_id": task_id},
                )

    async def _log_step(self, step_type: StepType, turn_number: int, data: dict[str, Any]) -> None:
        """Log a step event if a step logger is configured."""
        if self.step_logger:
            try:
                await self.step_logger.log_step(
                    StepEvent(step_type=step_type, turn_number=turn_number, data=data)
                )
            except Exception:
                logger.exception(
                    "Step logging failed",
                    extra={"step_type": step_type},
                )

    async def _log_assistant_blocks(self, content: list[Any], turn_number: int) -> None:
        """Log assistant message blocks (text and tool use) if step logger is configured."""
        if not self.step_logger:
            return
        for block in content:
            if isinstance(block, TextBlock):
                await self._log_step("assistant_message", turn_number, {"text": block.text})
            elif hasattr(block, "name") and hasattr(block, "input"):
                await self._log_step(
                    "tool_call",
                    turn_number,
                    {
                        "name": getattr(block, "name", ""),
                        "id": getattr(block, "id", "unknown"),
                        "input": getattr(block, "input", {}),
                    },
                )

    async def _log_turn_complete(self, message: ResultMessage, turn_number: int) -> None:
        """Log turn completion and emit telemetry."""
        await self._log_step(
            "turn_complete",
            turn_number,
            {
                "cost_usd": getattr(message, "total_cost_usd", None),
                "usage": getattr(message, "usage", None),
            },
        )
        await self._emit_telemetry(message)

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
        turn_number = 0

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                final_text = block.text
                        await self._log_assistant_blocks(message.content, turn_number)
                    elif isinstance(message, ResultMessage):
                        await self._log_turn_complete(message, turn_number)
                        if message.result:
                            final_text = message.result
                        structured_output = getattr(message, "structured_output", None)
                        turn_number += 1
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
        turn_number = 0

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        await self._log_assistant_blocks(message.content, turn_number)
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
                        await self._log_turn_complete(message, turn_number)
                        if message.result:
                            final_text = message.result
                        yield DoneEvent(
                            final_text=final_text,
                            structured_output=getattr(message, "structured_output", None),
                        )
                        turn_number += 1
                        break
        except Exception as e:
            logger.exception(
                "Claude Agent SDK streaming failed",
                extra={"prompt_preview": prompt[:100] if prompt else ""},
            )
            raise RuntimeError(f"Agent streaming failed: {type(e).__name__}: {e}") from e

"""Intermediate step logging for agent execution."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fastharness.logging import get_logger

logger = get_logger("step_logger")

StepType = Literal["tool_call", "assistant_message", "turn_complete"]


@dataclass
class StepEvent:
    """A single step event during execution."""

    step_type: StepType
    turn_number: int
    data: dict[str, Any]


class StepLogger(Protocol):
    """Protocol for step loggers."""

    async def log_step(self, event: StepEvent) -> None:
        """Log a single step event."""
        ...


class ConsoleStepLogger:
    """Logs steps to console using structured logging."""

    async def log_step(self, event: StepEvent) -> None:
        """Log step to console."""
        if event.step_type == "tool_call":
            logger.info(
                "Tool call",
                extra={
                    "turn": event.turn_number,
                    "tool_name": event.data.get("name"),
                    "tool_id": event.data.get("id"),
                },
            )
        elif event.step_type == "assistant_message":
            text_preview = event.data.get("text", "")[:100]
            logger.info(
                "Assistant message",
                extra={
                    "turn": event.turn_number,
                    "text_preview": text_preview,
                },
            )
        elif event.step_type == "turn_complete":
            logger.info(
                "Turn complete",
                extra={
                    "turn": event.turn_number,
                    "cost_usd": event.data.get("cost_usd"),
                    "tokens": event.data.get("usage"),
                },
            )

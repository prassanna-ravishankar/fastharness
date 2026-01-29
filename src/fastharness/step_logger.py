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
    """Logs steps to console with inline key=value formatting.

    Formats extra fields directly into the log message so they are
    visible with any logging formatter, including Python's default.
    """

    @staticmethod
    def _fmt(pairs: dict[str, Any]) -> str:
        """Format key=value pairs for log output."""
        return " | ".join(f"{k}={v}" for k, v in pairs.items() if v is not None)

    async def log_step(self, event: StepEvent) -> None:
        """Log step to console."""
        turn = event.turn_number
        if event.step_type == "tool_call":
            detail = self._fmt(
                {"turn": turn, "tool_name": event.data.get("name"), "tool_id": event.data.get("id")}
            )
            logger.info("Tool call | %s", detail)
        elif event.step_type == "assistant_message":
            text_preview = event.data.get("text", "")[:100]
            detail = self._fmt({"turn": turn, "text_preview": text_preview})
            logger.info("Assistant message | %s", detail)
        elif event.step_type == "turn_complete":
            detail = self._fmt(
                {
                    "turn": turn,
                    "cost_usd": event.data.get("cost_usd"),
                    "tokens": event.data.get("usage"),
                }
            )
            logger.info("Turn complete | %s", detail)
        else:
            detail = self._fmt({"step_type": event.step_type, "turn": turn})
            logger.warning("Unknown step type | %s", detail)

"""Streaming event types for agent execution."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TextEvent:
    """Text content event during streaming."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolEvent:
    """Tool use event during streaming."""

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_result: str | None = None
    type: Literal["tool"] = "tool"


@dataclass
class DoneEvent:
    """Completion event signaling end of execution."""

    final_text: str | None = None
    type: Literal["done"] = "done"


# Union type for all events
Event = TextEvent | ToolEvent | DoneEvent

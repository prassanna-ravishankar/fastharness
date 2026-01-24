"""Agent execution context."""

from dataclasses import dataclass, field
from typing import Any, Literal

# Role type for messages
RoleType = Literal["user", "assistant"]


@dataclass
class Message:
    """A message in the conversation history."""

    role: RoleType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Execution context passed to agent functions.

    Contains task information, conversation history, and injected dependencies.
    """

    task_id: str
    context_id: str
    message_history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    deps: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate context."""
        if not self.task_id:
            raise ValueError("AgentContext task_id cannot be empty")
        if not self.context_id:
            raise ValueError("AgentContext context_id cannot be empty")

    def get_last_user_message(self) -> str | None:
        """Get the most recent user message content."""
        for msg in reversed(self.message_history):
            if msg.role == "user":
                return msg.content
        return None

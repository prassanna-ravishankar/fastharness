"""Agent response types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    """An artifact produced by an agent (file, data, etc.)."""

    name: str
    mime_type: str = "text/plain"
    data: bytes | str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent execution."""

    text: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    data: dict[str, Any] | None = None

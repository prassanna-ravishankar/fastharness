"""Agent configuration and definition."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastharness.core.skill import Skill


@dataclass
class AgentConfig:
    """Configuration for a Claude agent.

    Contains all settings needed to create and run an agent.
    """

    name: str
    description: str
    skills: list[Skill]
    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)
    max_turns: int | None = None
    model: str = "claude-sonnet-4-20250514"
    custom_tools: list[Any] = field(default_factory=list)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    setting_sources: list[str] | None = field(default_factory=lambda: ["project"])
    output_format: dict[str, Any] | None = None
    tracing: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.name:
            raise ValueError("AgentConfig name cannot be empty")
        if not self.description:
            raise ValueError("AgentConfig description cannot be empty")
        if not self.skills:
            raise ValueError("AgentConfig must have at least one skill")
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("AgentConfig max_turns must be positive")


@dataclass
class Agent:
    """An agent with its configuration and optional custom loop function.

    If func is None, the agent uses default client.run() behavior.
    If func is provided, it controls the execution loop.
    """

    config: AgentConfig
    func: Callable[..., Any] | None = None

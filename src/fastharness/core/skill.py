"""A2A Skill metadata definition."""

from dataclasses import dataclass, field


@dataclass
class Skill:
    """A2A skill metadata for agent capabilities.

    Skills describe what an agent can do in A2A protocol terms.
    """

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    def __post_init__(self) -> None:
        """Validate skill configuration."""
        if not self.id:
            raise ValueError("Skill id cannot be empty")
        if not self.name:
            raise ValueError("Skill name cannot be empty")

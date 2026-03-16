"""AgentRuntime and AgentRuntimeFactory protocols — zero Claude SDK imports."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from fastharness.core.agent import AgentConfig
from fastharness.core.event import Event


@runtime_checkable
class AgentRuntime(Protocol):
    """Protocol for a single agent session that can execute prompts."""

    async def run(self, prompt: str) -> Any:
        """Execute a prompt and return the final result."""
        ...

    def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Execute a prompt, yielding events as they arrive."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by this runtime."""
        ...


@runtime_checkable
class AgentRuntimeFactory(Protocol):
    """Protocol for a factory that manages AgentRuntime lifecycle."""

    async def get_or_create(self, session_key: str, config: AgentConfig) -> AgentRuntime:
        """Return an existing runtime for session_key or create a new one."""
        ...

    async def remove(self, session_key: str) -> None:
        """Remove and close the runtime for session_key."""
        ...

    async def start_cleanup_task(self) -> None:
        """Start background cleanup of idle runtimes."""
        ...

    async def shutdown(self) -> None:
        """Shut down all runtimes and stop cleanup tasks."""
        ...

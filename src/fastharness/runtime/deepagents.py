"""Pydantic DeepAgents implementation of AgentRuntime and AgentRuntimeFactory."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastharness.core.agent import AgentConfig
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.logging import get_logger
from fastharness.runtime.base import BaseSessionFactory, SessionEntry

try:
    from pydantic_ai._agent_graph import CallToolsNode
    from pydantic_ai_backends import StateBackend
    from pydantic_deep import DeepAgentDeps, create_deep_agent
except ImportError as e:
    raise ImportError(
        "Pydantic DeepAgents is required for DeepAgentsRuntime. "
        "Install with: pip install fastharness[deepagents]"
    ) from e

logger = get_logger("runtime.deepagents")


@dataclass
class _DeepSession(SessionEntry):
    """DeepAgents-specific session entry."""

    agent: Any = field(default=None)  # pydantic_deep agent
    deps: Any = field(default=None)  # DeepAgentDeps
    message_history: list[Any] = field(default_factory=list)


def _create_agent(config: AgentConfig) -> Any:
    """Create a pydantic-deep agent from AgentConfig.

    Disables toolsets that eagerly validate API credentials at creation time
    (subagents, skills, web) — fastharness handles tool orchestration externally.
    """
    kwargs: dict[str, Any] = {
        "include_subagents": False,
        "include_skills": False,
        "include_web": False,
        "include_filesystem": False,
        "include_todo": False,
    }
    if config.model:
        kwargs["model"] = config.model
    if config.system_prompt:
        kwargs["instructions"] = config.system_prompt
    # Pass custom tools as a FunctionToolset to avoid create_deep_agent's
    # broken tool registration (it uses agent.tool() which requires RunContext).
    if config.custom_tools:
        from pydantic_ai import Tool
        from pydantic_ai.toolsets.function import FunctionToolset

        tools = [
            t if isinstance(t, Tool) else Tool(t, takes_ctx=False)
            for t in config.custom_tools
        ]
        kwargs["toolsets"] = [FunctionToolset(tools)]

    return create_deep_agent(**kwargs)


class DeepAgentsRuntime:
    """AgentRuntime backed by a Pydantic DeepAgent."""

    def __init__(self, agent: Any, deps: DeepAgentDeps, message_history: list[Any]) -> None:
        self._agent = agent
        self._deps = deps
        self._message_history = message_history

    async def run(self, prompt: str) -> Any:
        """Execute a prompt and return the output."""
        kwargs: dict[str, Any] = {"deps": self._deps}
        if self._message_history:
            kwargs["message_history"] = self._message_history

        result = await self._agent.run(prompt, **kwargs)

        # Preserve history for multi-turn (mutate in-place so factory's entry stays in sync)
        self._message_history[:] = result.all_messages()

        return result.output

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Stream events from agent execution."""
        kwargs: dict[str, Any] = {"deps": self._deps}
        if self._message_history:
            kwargs["message_history"] = self._message_history

        final_text: str | None = None

        async with self._agent.iter(prompt, **kwargs) as run:
            async for node in run:
                if isinstance(node, CallToolsNode):
                    if hasattr(node, "model_response") and hasattr(
                        node.model_response, "parts"
                    ):
                        for part in node.model_response.parts:
                            if hasattr(part, "tool_name"):
                                yield ToolEvent(
                                    tool_name=part.tool_name,
                                    tool_input=getattr(part, "args", {}),
                                )

            result = run.result
            if result:
                self._message_history[:] = result.all_messages()
                output = result.output
                if isinstance(output, str):
                    final_text = output
                    yield TextEvent(text=output)

        yield DoneEvent(final_text=final_text)

    async def aclose(self) -> None:
        """Release resources."""
        self._message_history.clear()


class DeepAgentsRuntimeFactory(BaseSessionFactory):
    """AgentRuntimeFactory that manages Pydantic DeepAgent sessions."""

    def __init__(self, ttl_minutes: int = 15) -> None:
        super().__init__(ttl_minutes=ttl_minutes, logger=logger)

    async def _create_session(self, config: AgentConfig, session_key: str = "") -> _DeepSession:
        agent = _create_agent(config)
        deps = DeepAgentDeps(backend=StateBackend())
        return _DeepSession(agent=agent, deps=deps)

    def _build_runtime(self, entry: SessionEntry) -> DeepAgentsRuntime:
        if not isinstance(entry, _DeepSession):
            raise TypeError(f"Expected _DeepSession, got {type(entry).__name__}")
        return DeepAgentsRuntime(entry.agent, entry.deps, entry.message_history)

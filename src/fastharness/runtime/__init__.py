"""AgentRuntime protocol and implementations.

Claude runtime is always available. OpenHands and DeepAgents runtimes
require their respective optional dependencies:
    pip install fastharness[openhands]
    pip install fastharness[deepagents]
"""

from fastharness.runtime.base import (
    AgentRuntime,
    AgentRuntimeFactory,
    BaseSessionFactory,
    SessionEntry,
)
from fastharness.runtime.claude import ClaudeRuntime, ClaudeRuntimeFactory

__all__ = [
    "AgentRuntime",
    "AgentRuntimeFactory",
    "BaseSessionFactory",
    "SessionEntry",
    "ClaudeRuntime",
    "ClaudeRuntimeFactory",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OpenHandsRuntime": ("fastharness.runtime.openhands", "OpenHandsRuntime"),
    "OpenHandsRuntimeFactory": ("fastharness.runtime.openhands", "OpenHandsRuntimeFactory"),
    "DeepAgentsRuntime": ("fastharness.runtime.deepagents", "DeepAgentsRuntime"),
    "DeepAgentsRuntimeFactory": ("fastharness.runtime.deepagents", "DeepAgentsRuntimeFactory"),
}


def __getattr__(name: str) -> type:
    """Lazy-load optional runtime implementations."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

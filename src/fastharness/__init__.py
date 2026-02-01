"""FastHarness - Wrap Claude Agent SDK and expose agents as A2A-compliant FastAPI services."""

from fastharness.app import FastHarness
from fastharness.client import HarnessClient
from fastharness.core.agent import Agent, AgentConfig
from fastharness.core.context import AgentContext, Message
from fastharness.core.event import DoneEvent, Event, TextEvent, ToolEvent
from fastharness.core.response import AgentResponse, Artifact
from fastharness.core.skill import Skill

__version__ = "1.0.0"

__all__ = [
    # Main class
    "FastHarness",
    # Client
    "HarnessClient",
    # Agent types
    "Agent",
    "AgentConfig",
    # Context
    "AgentContext",
    "Message",
    # Events
    "Event",
    "TextEvent",
    "ToolEvent",
    "DoneEvent",
    # Response
    "AgentResponse",
    "Artifact",
    # Skill
    "Skill",
]

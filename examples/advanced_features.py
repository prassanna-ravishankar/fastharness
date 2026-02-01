"""Advanced features example: tracing, MCP servers.

Demonstrates:
- OpenTelemetry tracing with GenAI semantic conventions
- MCP server configuration
- Custom agent loops

Run with: uv run uvicorn examples.advanced_features:app --port 8000
"""

import logging

from fastharness import (
    AgentContext,
    FastHarness,
    HarnessClient,
    Skill,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# Optional: set up OTel console exporter for development
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
except ImportError:
    pass  # OTel not installed, tracing will be a no-op

harness = FastHarness(
    name="advanced-features",
    description="Showcase tracing and MCP servers",
    version="1.0.0",
    tracing=True,
)

# Simple agent - inherits tracing=True from harness
harness.agent(
    name="simple-assistant",
    description="Basic assistant with tracing",
    skills=[
        Skill(
            id="assist",
            name="Assist",
            description="Provide assistance",
        )
    ],
    system_prompt="You are a helpful assistant. Provide concise responses.",
    tools=["Read", "Grep"],
    setting_sources=["project"],
)


# Custom loop with tracing
@harness.agentloop(
    name="tracked-researcher",
    description="Researcher with OTel tracing",
    skills=[
        Skill(
            id="research",
            name="Research",
            description="Conduct research with tracing",
            tags=["research", "tracing"],
        )
    ],
    system_prompt="You are a research assistant. Be thorough and structured.",
    tools=["Read", "WebSearch", "Glob"],
)
async def tracked_researcher(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
    """Research agent with OTel tracing.

    Each client.run() call creates a chat span with tool spans nested inside.
    The outer invoke_agent span wraps the entire execution.
    """
    result = await client.run(prompt)
    return result


# Custom loop with MCP servers (example config)
@harness.agentloop(
    name="mcp-agent",
    description="Agent with MCP server integration",
    skills=[
        Skill(
            id="integrated",
            name="Integrated",
            description="Use MCP servers for extended capabilities",
        )
    ],
    system_prompt="You are an agent with extended capabilities via MCP servers.",
    mcp_servers={},
    tools=["Read", "Glob"],
)
async def mcp_agent(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
    """Agent with MCP server support."""
    result = await client.run(prompt)
    return result


# Export the app
app = harness.app


if __name__ == "__main__":
    import uvicorn

    print("FastHarness Advanced Features Example")
    print("=" * 50)
    print("Agents with OTel tracing enabled")
    print("Starting server on http://localhost:8000")
    print("\nTest with:")
    print("  curl http://localhost:8000/.well-known/agent-card.json")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)

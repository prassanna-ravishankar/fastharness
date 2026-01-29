"""Advanced features example: telemetry, step logging, and MCP servers.

Demonstrates:
- Cost tracking with CostTracker
- Step logging with ConsoleStepLogger
- MCP server configuration
- Custom agent loops

Run with: uv run uvicorn examples.advanced_features:app --port 8000
"""

import logging

from fastharness import (
    AgentContext,
    ConsoleStepLogger,
    CostTracker,
    FastHarness,
    HarnessClient,
    Skill,
)

# Configure logging so fastharness telemetry/step output is visible
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

harness = FastHarness(
    name="advanced-features",
    description="Showcase cost tracking, step logging, and MCP servers",
    version="1.0.0",
)

# Initialize cost tracker
cost_tracker = CostTracker(warn_threshold_usd=0.5, error_threshold_usd=5.0)

# Simple agent - config only (no logging by default)
harness.agent(
    name="simple-assistant",
    description="Basic assistant without instrumentation",
    skills=[
        Skill(
            id="assist",
            name="Assist",
            description="Provide assistance",
        )
    ],
    system_prompt="You are a helpful assistant. Provide concise responses.",
    tools=["Read", "Grep"],
    setting_sources=["project"],  # Load CLAUDE.md if it exists
)


# Custom loop with cost tracking and step logging
@harness.agentloop(
    name="tracked-researcher",
    description="Researcher with cost tracking and detailed logging",
    skills=[
        Skill(
            id="research",
            name="Research",
            description="Conduct research with full telemetry",
            tags=["research", "tracking"],
        )
    ],
    system_prompt="You are a research assistant. Be thorough and structured.",
    tools=["Read", "WebSearch", "Glob"],
)
async def tracked_researcher(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
    """Research agent with cost tracking and step logging.

    When this agent runs, you'll see output like:
    [step_logger] Tool call: Read file.py
    [step_logger] Assistant message: Found bug...
    [step_logger] Turn complete: cost=$0.01, tokens=150

    Features:
    - Tracks API costs via telemetry callbacks
    - Logs each step (tool calls, messages, turn completion)
    - Automatically loads CLAUDE.md for project context
    """
    # Enable step logging
    client.enable_step_logging = True
    client.step_logger = ConsoleStepLogger()

    # Add cost tracking
    client.telemetry_callbacks.append(cost_tracker)

    # Run research with full instrumentation
    result = await client.run(prompt)

    return result


# Custom loop with MCP servers (example config, won't run without actual MCP server)
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
    mcp_servers={
        # Example MCP server configuration
        # Uncomment and configure to use actual MCP servers
        # "example-server": {
        #     "command": "python",
        #     "args": ["-m", "mcp_server_example"],
        # }
    },
    tools=["Read", "Glob"],  # Add "mcp__example-server__*" if MCP server is enabled
)
async def mcp_agent(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
    """Agent with MCP server support.

    MCP servers extend agent capabilities by connecting external services.
    Configure mcp_servers in the decorator to add custom tools.
    """
    client.telemetry_callbacks.append(cost_tracker)
    result = await client.run(prompt)
    return result


# Export the app
app = harness.app


if __name__ == "__main__":
    import uvicorn

    print("FastHarness Advanced Features Example")
    print("=" * 50)
    print("Agents with telemetry and step logging enabled")
    print("Starting server on http://localhost:8000")
    print("\nTest with:")
    print("  curl http://localhost:8000/.well-known/agent-card.json")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)

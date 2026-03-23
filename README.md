<p align="center">
  <img src="https://raw.githubusercontent.com/prassanna-ravishankar/fastharness/main/assets/logo.webp" alt="FastHarness" width="200">
</p>

# FastHarness

[![CI](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml/badge.svg)](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fastharness)](https://pypi.org/project/fastharness/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turn AI agents into production-ready A2A services — with pluggable runtime backends.**

FastHarness exposes agents through Google's [A2A (Agent-to-Agent)](https://a2a-protocol.org) protocol. Define agents with decorators, pick a runtime backend (Claude, OpenHands, Pydantic DeepAgents), or bridge existing agents (OpenClaw), and FastHarness handles protocol compliance, message conversion, task lifecycle, and multi-turn conversations.

```python
from fastharness import FastHarness, Skill

harness = FastHarness(name="my-agent")

harness.agent(
    name="assistant",
    description="A helpful assistant",
    skills=[Skill(id="help", name="Help", description="Answer questions")],
    system_prompt="You are helpful.",
    tools=["Read", "Grep"],
)

app = harness.app  # Ready to deploy
```

## Why FastHarness?

Building AI agents is easy. Making them **interoperable** is hard:

| Without FastHarness | With FastHarness |
|---------------------|------------------|
| Implement A2A protocol manually | Automatic A2A compliance |
| Handle message format conversion | Built-in message conversion |
| Manage task lifecycle and state | Managed task execution |
| Build conversation history tracking | **Multi-turn conversations out of the box** |
| Create JSON-RPC endpoints | FastAPI endpoints ready |
| Write agent card generation | Auto-generated agent cards |
| Lock into one agent framework | **Pluggable runtime backends** |

## What FastHarness Adds

On top of agent SDKs + A2A SDK:

- **Multi-turn conversations** — Runtime sessions maintain conversation history across A2A requests
- **Pluggable runtimes** — Swap between Claude, OpenHands, Pydantic DeepAgents, and OpenClaw backends
- **Cost tracking** — Built-in telemetry callbacks for monitoring API usage
- **Step logging** — Debug middleware for tool calls and intermediate steps
- **Zero-config protocol bridge** — Decorator API handles all A2A protocol machinery

## Installation

```bash
# Core (Claude Agent SDK backend)
uv add fastharness

# With OpenHands backend (requires Python 3.12+)
uv add fastharness[openhands]

# With Pydantic DeepAgents backend
uv add fastharness[deepagents]

# OpenClaw bridge (expose existing OpenClaw agents as A2A)
uv add fastharness[openclaw]

# All optional backends
uv add fastharness[openhands,deepagents,openclaw]
```

## Environment Setup

FastHarness automatically loads environment variables from a `.env` file in your project root at import time. Create one for API keys:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

The Claude runtime uses the Claude Agent SDK subprocess (which handles auth via the `claude` CLI). The DeepAgents and OpenHands runtimes use API keys directly — set `ANTHROPIC_API_KEY` in `.env` or your shell environment.

## Quick Start

**1. Define your agent:**

```python
from fastharness import FastHarness, Skill

harness = FastHarness(name="my-agent")

harness.agent(
    name="assistant",
    description="A helpful assistant",
    skills=[Skill(id="help", name="Help", description="Answer questions")],
    system_prompt="You are helpful.",
    tools=["Read", "Grep"],
)

app = harness.app
```

**2. Run it:**

```bash
uvicorn mymodule:app --port 8000
```

**3. Talk to it (Python):**

```python
import asyncio
from fastharness import FastHarnessClient

async def main():
    async with FastHarnessClient("http://localhost:8000") as client:
        reply = await client.send("Hello!")
        print(reply)

        # Multi-turn — same context_id maintains conversation
        reply = await client.send("My name is Alice", context_id="conv-1")
        reply = await client.send("What's my name?", context_id="conv-1")
        print(reply)  # "Alice"

        # Stream tokens as they arrive
        async for chunk in client.stream("Write a haiku"):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

<details>
<summary>Or with curl</summary>

```bash
# Agent card
curl http://localhost:8000/.well-known/agent-card.json

# Send a message
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello!"}],
        "messageId": "msg-1"
      }
    },
    "id": 1
  }'

# Stream (SSE)
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/sendStream",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello!"}],
        "messageId": "msg-1"
      }
    },
    "id": 1
  }'
```

</details>

## Runtime Backends

FastHarness uses an `AgentRuntime` / `AgentRuntimeFactory` protocol system that decouples agent execution from any specific SDK. You can swap backends without changing your agent definitions.

### Claude (default)

Uses the Claude Agent SDK via subprocess. No API key needed if the `claude` CLI is configured.

```python
from fastharness import FastHarness

harness = FastHarness(name="my-agent")  # ClaudeRuntimeFactory is the default
```

### OpenHands

Uses the [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk) for agents with terminal, file editing, and workspace capabilities.

```bash
uv add fastharness[openhands]
```

```python
from fastharness import FastHarness
from fastharness.runtime.openhands import OpenHandsRuntimeFactory

harness = FastHarness(
    name="dev-agent",
    runtime_factory=OpenHandsRuntimeFactory(workspace="/path/to/project"),
)
```

### Pydantic DeepAgents

Uses [Pydantic DeepAgents](https://github.com/vstorm-co/pydantic-deepagents) for agents built on pydantic-ai with planning, subagents, and structured output.

```bash
uv add fastharness[deepagents]
```

```python
from fastharness import FastHarness
from fastharness.runtime.deepagents import DeepAgentsRuntimeFactory

harness = FastHarness(
    name="research-agent",
    runtime_factory=DeepAgentsRuntimeFactory(),
)
```

**Note:** DeepAgents requires `ANTHROPIC_API_KEY` (or the appropriate provider key) in your environment or `.env` file.

### OpenClaw Bridge

Already running agents on [OpenClaw](https://openclaw.ai)? Expose them as A2A endpoints without rewriting anything:

```bash
uv add fastharness[openclaw]
```

```python
from fastharness.bridges.openclaw import OpenClawBridge

bridge = OpenClawBridge("ws://localhost:18789")  # your OpenClaw gateway

# One-liner: single agent
app = bridge.expose("research-bot", description="Research assistant")

# Multiple agents on one service
harness = bridge.to_harness("my-agents")
bridge.add_agent(harness, "research-bot", description="Research")
bridge.add_agent(harness, "coder-bot", description="Coding assistant")
app = harness.app
```

Your OpenClaw agents get A2A protocol compliance, streaming, multi-turn conversations, and health endpoints — zero changes to the agents themselves.

### Custom Runtimes

Implement the `AgentRuntime` and `AgentRuntimeFactory` protocols to add your own backend:

```python
from fastharness.runtime.base import AgentRuntime, AgentRuntimeFactory

class MyRuntime:
    async def run(self, prompt: str) -> Any:
        """Execute prompt, return result."""
        ...

    async def stream(self, prompt: str) -> AsyncIterator[Event]:
        """Execute prompt, yield events."""
        ...

    async def aclose(self) -> None:
        """Cleanup resources."""
        ...

class MyRuntimeFactory:
    async def get_or_create(self, session_key: str, config: AgentConfig) -> MyRuntime:
        ...
    async def remove(self, session_key: str) -> None:
        ...
    async def start_cleanup_task(self) -> None:
        ...
    async def shutdown(self) -> None:
        ...

harness = FastHarness(name="my-agent", runtime_factory=MyRuntimeFactory())
```

## Multi-Turn Conversations

FastHarness maintains conversation history automatically. Just use the same `contextId` on the message — this is the standard A2A conversation identifier:

```python
# Message 1: "My name is Alice"
# → Response: "Nice to meet you, Alice!"

# Message 2: "What's my name?" (same contextId)
# → Response: "Your name is Alice!"
```

**How it works:**
- All messages with the same `contextId` share history via the runtime session pool
- Sessions are reused for 15 minutes by default (configurable via `ttl_minutes`)
- No manual history management needed

<details>
<summary>Full example with curl</summary>

```bash
# First message
curl -X POST http://localhost:8000/ -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "contextId": "conv-123",
      "parts": [{"kind": "text", "text": "My name is Alice"}],
      "messageId": "msg-1"
    }
  },
  "id": 1
}'

# Follow-up (agent remembers "Alice")
curl -X POST http://localhost:8000/ -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "contextId": "conv-123",
      "parts": [{"kind": "text", "text": "What is my name?"}],
      "messageId": "msg-2"
    }
  },
  "id": 2
}'
```

</details>

## Usage Patterns

### Custom Agent Loop

Take full control over execution while FastHarness handles protocol machinery:

```python
@harness.agentloop(
    name="researcher",
    description="Deep research assistant",
    skills=[Skill(id="research", name="Research", description="Conduct research")],
)
async def researcher(prompt, ctx, client):
    """Custom loop with iterative refinement."""
    result = await client.run(prompt)

    # Keep researching until confident
    while "need more information" in result.lower():
        result = await client.run("Continue researching, go deeper")

    return result
```

### Mount on Existing FastAPI App

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    async with harness.lifespan_context():
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/agents", harness.app)
```

## Advanced Features

<details>
<summary><b>Cost Tracking</b></summary>

Monitor API costs with configurable thresholds:

```python
from fastharness import CostTracker

tracker = CostTracker(warn_threshold_usd=1.0)

@harness.agentloop(...)
async def agent(prompt, ctx, client):
    client.telemetry_callbacks.append(tracker)
    result = await client.run(prompt)
    print(f"Cost: ${tracker.total_cost_usd:.4f}")
    return result
```

</details>

<details>
<summary><b>Step Logging</b></summary>

Debug with detailed step-by-step logging:

```python
from fastharness import ConsoleStepLogger

client = HarnessClient(step_logger=ConsoleStepLogger())
result = await client.run(prompt)
```

Output:
```
[step_logger] Tool call
    turn: 0
    tool_name: Read
    tool_id: call_123
[step_logger] Assistant message
    turn: 0
    text_preview: Found the bug...
[step_logger] Turn complete
    turn: 0
    cost_usd: 0.01
```

</details>

<details>
<summary><b>MCP Server Integration</b></summary>

Connect external services via Model Context Protocol:

```python
harness.agent(
    name="assistant",
    description="Multi-tool assistant",
    skills=[...],
    mcp_servers={
        "filesystem": {
            "command": "node",
            "args": ["mcp-server-stdio-filesystem"],
        },
    },
    tools=["mcp__filesystem__read", "mcp__filesystem__write"],
)
```

</details>

<details>
<summary><b>CLAUDE.md Support</b></summary>

Agents automatically load project context from `CLAUDE.md`:

```python
harness.agent(
    name="reviewer",
    description="Code reviewer",
    skills=[...],
    # setting_sources=["project"] is default - loads CLAUDE.md
)

# To disable:
harness.agent(
    name="readonly",
    description="Read-only assistant",
    skills=[...],
    setting_sources=[],  # Don't load settings
)
```

</details>

## Configuration

### HarnessClient Options

| Option | Default | Description |
|--------|---------|-------------|
| `system_prompt` | `None` | System prompt for the agent |
| `tools` | `[]` | Allowed tools (`["Read", "Grep", "Glob"]`) |
| `model` | `claude-sonnet-4-20250514` | Model identifier |
| `max_turns` | `None` | Max conversation turns |
| `mcp_servers` | `{}` | MCP server configs |
| `setting_sources` | `["project"]` | Load CLAUDE.md automatically |
| `output_format` | `None` | JSON schema for structured output |
| `runtime` | `None` | Injected `AgentRuntime` (set by factory) |

Override per-call:
```python
result = await client.run(prompt, model="claude-opus-4-20250514", max_turns=5)
```

### A2A Endpoints

| Endpoint | Description |
|----------|-------------|
| `/.well-known/agent-card.json` | Agent metadata and capabilities |
| `/` | JSON-RPC endpoint (`message/send`, `message/sendStream`, `tasks/get`, `tasks/cancel`) |
| `/docs` | Interactive API documentation |

## Architecture

```
FastHarness (app.py)
    │
    ├── Decorators: .agent() and .agentloop()
    │   └── Register agents with skills, tools, system prompts
    │
    ├── A2AFastAPIApplication (native A2A SDK)
    │   └── Exposes A2A endpoints: /.well-known/agent-card.json, JSON-RPC /
    │
    ├── ClaudeAgentExecutor (worker/claude_executor.py)
    │   └── Executes tasks using HarnessClient → AgentRuntime
    │
    └── AgentRuntimeFactory (runtime/base.py)  ← Protocol
        ├── ClaudeRuntimeFactory     → Claude Agent SDK subprocess
        ├── OpenHandsRuntimeFactory  → OpenHands SDK Conversation
        ├── DeepAgentsRuntimeFactory → Pydantic DeepAgents
        └── OpenClawRuntimeFactory   → OpenClaw gateway (via bridge)
```

**Flow**: A2A request → DefaultRequestHandler → ClaudeAgentExecutor → HarnessClient → AgentRuntime → SDK → A2A response

## Examples

Complete working examples in [`examples/`](examples/):

- **[simple_agent.py](examples/simple_agent.py)** - Basic agent with multi-turn support
- **[fastapi_integration.py](examples/fastapi_integration.py)** - Mounting on existing FastAPI app
- **[advanced_features.py](examples/advanced_features.py)** - Cost tracking, logging, MCP servers

Run examples:
```bash
uv run uvicorn examples.simple_agent:app --port 8000
```

## LiteLLM Integration

Route through LiteLLM for alternative providers:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=your-litellm-key
ANTHROPIC_MODEL=sonnet-4
```

## License

MIT © [Prassanna Ravishankar](https://github.com/prassanna-ravishankar)

---

<p align="center">
  <i>Built for the AI agent community</i>
</p>

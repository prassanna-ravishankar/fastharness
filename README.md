<p align="center">
  <img src="https://raw.githubusercontent.com/prassanna-ravishankar/fastharness/main/assets/logo.webp" alt="FastHarness" width="200">
</p>

# FastHarness

[![CI](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml/badge.svg)](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fastharness)](https://pypi.org/project/fastharness/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turn Claude Agent SDK agents into production-ready A2A services in minutes.**

FastHarness wraps the Claude Agent SDK and automatically exposes your agents through Google's [A2A (Agent-to-Agent)](https://a2a-protocol.org) protocol. Define agents with decorators, and FastHarness handles protocol compliance, message conversion, task lifecycle, and multi-turn conversations.

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

Building Claude agents is easy. Making them **interoperable** is hard:

| Without FastHarness | With FastHarness |
|---------------------|------------------|
| Implement A2A protocol manually | Automatic A2A compliance |
| Handle message format conversion | Built-in message conversion |
| Manage task lifecycle and state | Managed task execution |
| Build conversation history tracking | **Multi-turn conversations out of the box** |
| Create JSON-RPC endpoints | FastAPI endpoints ready |
| Write agent card generation | Auto-generated agent cards |

## What FastHarness Adds

On top of Claude Agent SDK + A2A SDK:

- **Multi-turn conversations** - Client pooling maintains conversation history across A2A requests
- **Cost tracking** - Built-in telemetry callbacks for monitoring API usage
- **Step logging** - Debug middleware for tool calls and intermediate steps
- **Zero-config protocol bridge** - Decorator API handles all A2A protocol machinery

## Installation

```bash
uv add fastharness
```

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

**3. Test it:**

```bash
# Get agent card
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
      },
      "metadata": {"conversation_id": "conv-123"}
    },
    "id": 1
  }'
```

## Multi-Turn Conversations

FastHarness maintains conversation history automatically. Just use the same `conversation_id`:

```python
# Message 1: "My name is Alice"
# → Response: "Nice to meet you, Alice!"

# Message 2: "What's my name?" (same conversation_id)
# → Response: "Your name is Alice!"
```

**How it works:**
- All messages with the same `conversation_id` in metadata share history
- Claude SDK clients are pooled and reused for 15 minutes
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
      "parts": [{"kind": "text", "text": "My name is Alice"}],
      "messageId": "msg-1"
    },
    "metadata": {"conversation_id": "conv-123"}
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
      "parts": [{"kind": "text", "text": "What is my name?"}],
      "messageId": "msg-2"
    },
    "metadata": {"conversation_id": "conv-123"}
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
| `system_prompt` | `None` | System prompt for Claude |
| `tools` | `[]` | Allowed tools (`["Read", "Grep", "Glob"]`) |
| `model` | `claude-sonnet-4-20250514` | Claude model |
| `max_turns` | `None` | Max conversation turns |
| `mcp_servers` | `{}` | MCP server configs |
| `setting_sources` | `["project"]` | Load CLAUDE.md automatically |
| `output_format` | `None` | JSON schema for structured output |

Override per-call:
```python
result = await client.run(prompt, model="claude-opus-4-20250514", max_turns=5)
```

### A2A Endpoints

| Endpoint | Description |
|----------|-------------|
| `/.well-known/agent-card.json` | Agent metadata and capabilities |
| `/` | JSON-RPC endpoint (`message/send`, `tasks/get`) |
| `/docs` | Interactive API documentation |

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
  <i>Built for the Claude Agent SDK community</i>
</p>

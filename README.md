# FastHarness

[![CI](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml/badge.svg)](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Building agents with the Claude SDK is straightforward, but exposing them as interoperable services requires implementing protocol layers, managing task lifecycles, and handling message conversion between formats. FastHarness bridges this gap by wrapping the Claude Agent SDK and automatically exposing your agents through Google's [A2A (Agent-to-Agent)](https://github.com/google/A2A) protocol.

The library provides a decorator-based API where you define agent behavior and FastHarness handles the rest: generating agent cards, exposing JSON-RPC endpoints, converting between Claude SDK messages and A2A format, and managing async task execution. A simple agent requires only a name, description, and list of skills. For complex workflows that need multi-turn reasoning or custom control flow, the `@agentloop` decorator gives you full control over the execution loop while FastHarness manages the protocol machinery.

FastHarness runs standalone or mounts onto existing FastAPI applications, making it suitable for both dedicated agent services and adding agent capabilities to existing APIs. The underlying Claude SDK calls can be routed through LiteLLM, enabling use of alternative model providers without code changes.

## Installation

```bash
uv add fastharness
```

## Quick Start

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

Run with:
```bash
uvicorn mymodule:app --port 8000
```

Test:
```bash
curl http://localhost:8000/.well-known/agent-card.json
```

## Custom Agent Loop

```python
@harness.agentloop(
    name="researcher",
    description="Multi-turn researcher",
    skills=[Skill(id="research", name="Research", description="Deep research")],
)
async def researcher(prompt, ctx, client):
    result = await client.run(prompt)
    while "need more" in result.lower():
        result = await client.run("Continue researching")
    return result
```

## Mount on FastAPI

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

## HarnessClient Options

The `HarnessClient` passed to agent functions supports these options:

| Option | Default | Description |
|--------|---------|-------------|
| `system_prompt` | `None` | System prompt for Claude |
| `tools` | `[]` | Allowed tools (e.g., `["Read", "Grep", "Glob"]`) |
| `model` | `claude-sonnet-4-20250514` | Claude model to use |
| `max_turns` | `None` | Maximum conversation turns |
| `permission_mode` | `bypassPermissions` | Permission handling mode |

Override per-call:
```python
result = await client.run(prompt, model="claude-opus-4-20250514", max_turns=5)
```

## A2A Endpoints

Running FastHarness exposes these endpoints:

| Endpoint | Description |
|----------|-------------|
| `/.well-known/agent-card.json` | Agent metadata and capabilities |
| `/` | JSON-RPC endpoint (`message/send`, `tasks/get`, etc.) |
| `/docs` | Interactive documentation |

## LiteLLM Support

Set environment variables to use LiteLLM as a proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=your-litellm-key
ANTHROPIC_MODEL=sonnet-4
```

## License

MIT

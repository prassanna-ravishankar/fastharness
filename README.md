# FastHarness

[![CI](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml/badge.svg)](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Wrap Claude Agent SDK and expose agents as [A2A](https://github.com/google/A2A)-compliant services.

## Features

- **A2A Protocol** - Full compliance with Google's Agent-to-Agent protocol
- **Simple API** - Decorator-based agent registration with `@agentloop`
- **Multi-turn Support** - Custom execution loops for complex agent workflows
- **FastAPI Integration** - Mount on existing FastAPI apps or run standalone
- **LiteLLM Compatible** - Use any LLM provider via LiteLLM proxy

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

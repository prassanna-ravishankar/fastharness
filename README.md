<p align="center">
  <img src="https://raw.githubusercontent.com/prassanna-ravishankar/fastharness/main/assets/logo.webp" alt="FastHarness" width="200">
</p>

# FastHarness

[![CI](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml/badge.svg)](https://github.com/prassanna-ravishankar/fastharness/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fastharness)](https://pypi.org/project/fastharness/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Building agents with the Claude Agent SDK is straightforward, but exposing them as interoperable services requires implementing protocol layers, managing task lifecycles, and handling message conversion between formats. FastHarness bridges this gap by wrapping the Claude Agent SDK and automatically exposing your agents through Google's [A2A (Agent-to-Agent)](https://a2a-protocol.org) protocol.

Define agents with a simple decorator-based API. FastHarness handles the rest: generating agent cards, exposing JSON-RPC endpoints, converting between Claude SDK messages and A2A format, and managing async task execution. A simple agent requires only a name, description, and list of skills. For complex workflows, the `@agentloop` decorator gives you full control over the execution loop while FastHarness manages the protocol machinery.

Features include:
- **Tracing** - OpenTelemetry GenAI semantic conventions for multi-turn agent tracing (optional)
- **CLAUDE.md support** - Automatically load project context and conventions
- **MCP servers** - Connect external services via Model Context Protocol
- **FastAPI integration** - Run standalone or mount on existing applications
- **LiteLLM support** - Route API calls through alternative providers

FastHarness is production-ready and fully A2A protocol compliant.

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

## v1.0.0 Features

### CLAUDE.md Support

Agents automatically load project settings from `CLAUDE.md` by default:

```python
harness.agent(
    name="reviewer",
    description="Code reviewer",
    skills=[...],
    system_prompt="You are a thorough code reviewer.",
    # setting_sources=["project"] is default - loads CLAUDE.md automatically
)

# To disable, pass empty list:
harness.agent(
    name="readonly",
    description="Read-only assistant",
    skills=[...],
    setting_sources=[],  # Don't load any settings
)
```

### MCP Server Integration

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

### Tracing

FastHarness supports [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
for distributed tracing and metrics. Install the optional dependency and enable tracing:

```bash
uv add fastharness[otel]
```

```python
harness = FastHarness(name="my-agent", tracing=True)
harness.agent(name="assistant", description="...", skills=[...])
app = harness.app
```

Configure your OTel pipeline separately (standard OTel setup):

<details>
<summary>Console exporter (development)</summary>

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

harness = FastHarness(name="my-agent", tracing=True)
harness.agent(name="assistant", description="...", skills=[...])
app = harness.app
```

</details>

<details>
<summary>OTLP exporter (production)</summary>

```bash
uv add opentelemetry-exporter-otlp opentelemetry-sdk
export OTEL_SERVICE_NAME=my-agent
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

harness = FastHarness(name="my-agent", tracing=True)
harness.agent(name="assistant", description="...", skills=[...])
app = harness.app
```

Or use auto-instrumentation:
```bash
opentelemetry-instrument uvicorn mymodule:app --port 8000
```

</details>

<details>
<summary>Jaeger (local debugging)</summary>

```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")))
trace.set_tracer_provider(provider)

harness = FastHarness(name="my-agent", tracing=True)
harness.agent(name="assistant", description="...", skills=[...])
app = harness.app
# View traces at http://localhost:16686
```

</details>

<details>
<summary>Span hierarchy</summary>

A multi-turn agent execution with tool use:

```
invoke_agent assistant                    (850ms)
  └── chat claude-sonnet-4-20250514       (845ms)
        ├── execute_tool Read             (toolu_abc)
        ├── execute_tool Grep             (toolu_def)
        └── execute_tool Write            (toolu_ghi)
```

</details>

<details>
<summary>Multi-step agentloop</summary>

Custom agent loops that call `client.run()` multiple times get separate chat spans:

```python
@harness.agentloop(name="researcher", ..., tracing=True)
async def researcher(prompt, ctx, client):
    draft = await client.run(prompt)            # → chat span 1
    review = await client.run(f"Review: {draft}")  # → chat span 2
    return review
```

```
invoke_agent researcher                   (2400ms)
  ├── chat claude-sonnet-4-...            (1800ms)
  │     ├── execute_tool Read
  │     └── execute_tool Grep
  └── chat claude-sonnet-4-...            (600ms)
```

</details>

<details>
<summary>Per-agent tracing</summary>

```python
harness = FastHarness(name="my-service", tracing=True)

# Inherits tracing=True from harness
harness.agent(name="traced-agent", description="...", skills=[...])

# Explicitly disable for this agent
harness.agent(name="silent-agent", description="...", skills=[...], tracing=False)
```

</details>

## HarnessClient Options

The `HarnessClient` passed to agent functions supports these options:

| Option | Default | Description |
|--------|---------|-------------|
| `system_prompt` | `None` | System prompt for Claude |
| `tools` | `[]` | Allowed tools (e.g., `["Read", "Grep", "Glob"]`) |
| `model` | `claude-sonnet-4-20250514` | Claude model to use |
| `max_turns` | `None` | Maximum conversation turns |
| `permission_mode` | `bypassPermissions` | Permission handling mode |
| `mcp_servers` | `{}` | MCP server configurations |
| `setting_sources` | `["project"]` | Filesystem settings to load (loads CLAUDE.md) |
| `output_format` | `None` | JSON schema for structured output (e.g., `{"type": "json_schema", "schema": {...}}`) |
| `tracing` | `False` | Enable OpenTelemetry tracing |

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

## Examples

See the `examples/` directory for complete working examples:

- **[simple_agent.py](examples/simple_agent.py)** - Standalone agent service with multi-agent support
- **[fastapi_integration.py](examples/fastapi_integration.py)** - Mounting FastHarness on existing FastAPI app
- **[advanced_features.py](examples/advanced_features.py)** - Cost tracking, step logging, and MCP server configuration

Run examples:
```bash
# Standalone agent
uv run uvicorn examples.simple_agent:app --port 8000

# FastAPI integration
uv run uvicorn examples.fastapi_integration:app --port 8000
```

Test with:
```bash
# Get agent card
curl http://localhost:8000/.well-known/agent-card.json

# Send message (A2A JSON-RPC)
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "id": "task-001",
      "contextId": "ctx-001",
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello"}],
        "kind": "message",
        "messageId": "msg-001"
      }
    },
    "id": 1
  }'
```

## LiteLLM Support

Set environment variables to use LiteLLM as a proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=your-litellm-key
ANTHROPIC_MODEL=sonnet-4
```

## License

MIT

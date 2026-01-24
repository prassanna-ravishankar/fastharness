# FastHarness

Wrap Claude Agent SDK and expose agents as [A2A](https://github.com/google/A2A)-compliant services.

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

## LiteLLM Support

Set environment variables to use LiteLLM as a proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=your-litellm-key
ANTHROPIC_MODEL=sonnet-4
```

## License

MIT

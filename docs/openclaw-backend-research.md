# OpenClaw as Backend Runtime — Research Notes

**Status:** Research complete, pivoted to upstream direction (fastharness as framework for OpenClaw).

## OpenClaw SDK Summary

- **Package:** `openclaw-sdk` on PyPI (Python 3.11+)
- **Architecture:** Connects to OpenClaw gateway via WebSocket RPC
- **Async:** Fully native async — no sync bridging needed

## Key API

```python
from openclaw_sdk import OpenClawClient

async with OpenClawClient.connect() as client:
    agent = client.get_agent("agent-id")

    # Single-shot
    result = await agent.execute("query")  # → ExecutionResult
    print(result.content, result.token_usage)

    # Streaming (typed)
    async for event in await agent.execute_stream_typed("query"):
        # ContentEvent, ToolCallEvent, DoneEvent, ErrorEvent
        ...

    # Multi-turn
    conv = agent.conversation("session-1")
    await conv.say("message 1")
    await conv.say("message 2")  # server-side history
```

## Proposed AgentRuntime Mapping

| AgentRuntime | OpenClaw SDK |
|---|---|
| `run(prompt)` | `conversation.say(prompt)` → `.content` |
| `stream(prompt)` | `execute_stream_typed(prompt)` → ContentEvent/ToolCallEvent/DoneEvent |
| `aclose()` | WebSocket client close |
| Session persistence | `agent.conversation(session_name)` |

## TypedStreamEvent → FastHarness Event Mapping

| OpenClaw | FastHarness |
|---|---|
| `ContentEvent(text=...)` | `TextEvent(text=...)` |
| `ToolCallEvent(tool=..., input=...)` | `ToolEvent(tool_name=..., tool_input=...)` |
| `DoneEvent(content=..., token_usage=...)` | `DoneEvent(final_text=..., metrics={...})` |
| `ErrorEvent(message=...)` | raise RuntimeError |

## Design

- `OpenClawRuntimeFactory` extends `BaseSessionFactory`
- One `OpenClawClient.connect()` per session
- Gateway URL configurable or auto-detected from `OPENCLAW_GATEWAY_WS_URL`
- Optional dep: `pip install fastharness[openclaw]`

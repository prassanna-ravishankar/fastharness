# Migration Guide: v0.x → v1.0.0

## Overview

FastHarness v1.0.0 is a major upgrade that migrates from `claude-code-sdk` to the official `claude-agent-sdk` and introduces four key features: CLAUDE.md support, MCP server support, cost tracking/telemetry, and intermediate step logging.

## Breaking Changes

### 1. SDK Dependency

**Old**: `claude-code-sdk>=0.0.20`
**New**: `claude-agent-sdk>=0.1.0`

If you're only using the FastHarness API (decorators, client), no code changes needed. If you import SDK types directly, update as below.

### 2. Direct SDK Type Imports

**Before:**
```python
from claude_code_sdk import ClaudeCodeOptions
```

**After:**
```python
from claude_agent_sdk import ClaudeAgentOptions
```

FastHarness exports `HarnessClient` which handles this internally, so most applications won't be affected.

### 3. Default Settings Behavior

The new SDK doesn't load filesystem settings by default, but FastHarness loads project settings (`CLAUDE.md`, etc.) by default for convenience.

**Old behavior:** Settings were loaded automatically
**New behavior:** Explicitly configured via `setting_sources` (defaults to `["project"]` in FastHarness)

To disable CLAUDE.md loading:
```python
harness.agent(
    name="assistant",
    description="...",
    skills=[...],
    setting_sources=[],  # Don't load any settings
)
```

### 4. System Prompt Handling

The new SDK requires explicit system prompts. FastHarness continues to support optional prompts that merge with CLAUDE.md.

**Old:** System prompt defaulted to Claude Code's built-in prompt
**New:** Explicit prompts only (CLAUDE.md + agent's system_prompt if specified)

## New Features

### 1. CLAUDE.md Support

Agents automatically load `CLAUDE.md` from your project by default, merging it with the agent's system prompt.

```python
harness.agent(
    name="reviewer",
    description="Code reviewer",
    skills=[...],
    system_prompt="You are a thorough code reviewer.",  # Agent identity
    # setting_sources=["project"] is default
)

# To disable:
harness.agent(
    name="isolated",
    description="No project context",
    skills=[...],
    setting_sources=[],  # Don't load CLAUDE.md
)
```

**What loads:** CLAUDE.md from the project root (merged into system prompt)
**Order:** Agent's system_prompt (prepend) + CLAUDE.md (append)
**Default:** `setting_sources=["project"]` - automatically enabled

### 2. MCP Server Support

Connect external services via Model Context Protocol.

```python
harness.agent(
    name="assistant",
    description="Multi-tool assistant",
    skills=[...],
    mcp_servers={
        "filesystem": {
            "command": "node",
            "args": ["server.js"],
        },
        "github": {
            "command": "uvx",
            "args": ["mcp-server-github"],
        },
    },
    tools=["mcp__filesystem__read", "mcp__github__search", ...],
)
```

MCP tools appear in the `allowed_tools` list with prefix: `mcp__<server>__<tool>`.

### 3. Cost Tracking & Telemetry

Track execution costs with configurable thresholds.

```python
from fastharness import CostTracker

tracker = CostTracker(
    warn_threshold_usd=1.0,
    error_threshold_usd=10.0,
)

@harness.agentloop(name="researcher", ...)
async def researcher(prompt, ctx, client):
    client.telemetry_callbacks.append(tracker)
    result = await client.run(prompt)
    print(f"Total cost: ${tracker.total_cost_usd:.4f}")
    print(f"Executions: {len(tracker.executions)}")
    return result
```

**Metrics captured:**
- `total_cost_usd`: Total API cost
- `input_tokens`, `output_tokens`: Token usage
- `cache_read_tokens`, `cache_write_tokens`: Prompt caching
- `duration_ms`: Total execution time
- `num_turns`: Agent turns
- `status`: "success" or "error"

### 4. Step Logging

Log intermediate steps (tool calls, messages, turn completions) during execution.

```python
from fastharness import ConsoleStepLogger

client = HarnessClient(
    enable_step_logging=True,
    step_logger=ConsoleStepLogger(),
)

result = await client.run(prompt)
# Output:
# [turn 0] Tool call: Read file.py
# [turn 0] Assistant message: Found the bug...
# [turn 0] Turn complete: cost=$0.01, tokens=150
```

**Custom logger:**
```python
class MyStepLogger:
    async def log_step(self, event: StepEvent) -> None:
        if event.step_type == "tool_call":
            print(f"Calling {event.data['name']}")
        elif event.step_type == "assistant_message":
            print(f"Assistant: {event.data['text'][:50]}...")
```

## Migration Checklist

- [ ] Update `pyproject.toml`: `claude-code-sdk` → `claude-agent-sdk>=0.1.0`
- [ ] Run `uv sync` to install new dependency
- [ ] Update direct SDK imports (if any):
  - `ClaudeCodeOptions` → `ClaudeAgentOptions`
- [ ] Test existing agents (should work unchanged - CLAUDE.md loads by default via `setting_sources=["project"]`)
- [ ] Add telemetry to monitor costs (optional but recommended)
- [ ] Add step logging for debugging (optional)
- [ ] Create `CLAUDE.md` in project root for agent workflows (optional)

## Backward Compatibility

### What Still Works

- Simple agent registration via `harness.agent()`
- Custom loops via `@harness.agentloop()`
- `HarnessClient.run()` and `HarnessClient.stream()`
- Agent context and skills
- A2A protocol compliance

### What Changed

- SDK package name and types
- System prompt defaults (now requires explicit prompt)
- Settings loading (now opt-in via `load_claude_md`)
- MCP configuration (now explicit, was hardcoded to `{}`)

## Examples

### Simple Agent (Config-only)

```python
from fastharness import FastHarness, Skill

harness = FastHarness()

harness.agent(
    name="helper",
    description="Helpful assistant",
    skills=[Skill(id="help", name="Help", description="Answer questions")],
    system_prompt="You are a helpful assistant.",
    tools=["Read", "Glob", "Grep"],
    load_claude_md=True,  # Load CLAUDE.md for project context
)

app = harness.app
```

### Agent with MCP + Telemetry

```python
from fastharness import FastHarness, Skill, CostTracker, ConsoleStepLogger

harness = FastHarness()
tracker = CostTracker(warn_threshold_usd=0.5)

@harness.agentloop(
    name="researcher",
    description="Research assistant with web access",
    skills=[Skill(id="research", name="Research", description="Find information")],
    mcp_servers={
        "web": {"command": "uvx", "args": ["mcp-server-fetch"]},
    },
    tools=["mcp__web__fetch", "WebSearch", "Read"],
    load_claude_md=True,
)
async def researcher(prompt, ctx, client):
    client.telemetry_callbacks.append(tracker)
    client.enable_step_logging = True
    client.step_logger = ConsoleStepLogger()

    result = await client.run(prompt)
    print(f"Cost: ${tracker.total_cost_usd:.4f}")
    return result

app = harness.app
```

### CLAUDE.md Project Context

Create `CLAUDE.md` in your project root:

```markdown
# Project Conventions

- Use type hints for all functions
- Follow PEP 8 style guide
- Write docstrings for public APIs
- Use descriptive variable names
```

Agent system prompts will automatically include this.

## Troubleshooting

### "Module not found: claude_agent_sdk"

Install the new dependency:
```bash
uv sync --upgrade
# or
pip install claude-agent-sdk>=0.1.0
```

### "CLAUDE.md not being loaded"

Ensure:
1. `CLAUDE.md` exists in your project root
2. `load_claude_md=True` (default) is set on the agent
3. Run from the project directory

### "MCP tools not available"

Check:
1. MCP server command is correct
2. Tool names have `mcp__` prefix in `allowed_tools`
3. MCP server is installed and executable

### "Telemetry not recording"

Ensure:
1. `client.telemetry_callbacks.append(tracker)` is called before `client.run()`
2. Callback is async-compatible
3. No exceptions during execution

## Support

For issues migrating to v1.0.0:
1. Check the examples in the FastHarness repository
2. Review claude-agent-sdk docs: https://platform.claude.com/docs/en/agent-sdk/overview
3. File an issue on GitHub with migration details

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastHarness exposes AI agents as A2A (Agent-to-Agent) protocol compliant services with pluggable runtime backends (Claude Agent SDK, OpenHands, Pydantic DeepAgents). It bridges agent SDKs with Google's A2A protocol using the native A2A Python SDK.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest
uv run pytest tests/test_basic.py::TestSkill  # Single test class
uv run pytest -k "test_agent"                  # Pattern match

# Type check
uv run ty check src/

# Lint
uv run ruff check src/
uv run ruff format src/  # Format code
```

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

### Key Components

- **FastHarness** (`app.py`): Main entry point. Registers agents, creates A2AFastAPIApplication with native SDK integration. Accepts optional `runtime_factory` for backend selection.
- **HarnessClient** (`client.py`): Delegates to an `AgentRuntime`. Provides `run()` for full execution and `stream()` for event-based iteration.
- **AgentRuntime / AgentRuntimeFactory** (`runtime/base.py`): Protocols that decouple execution from any specific SDK. Implementations in `runtime/claude.py`, `runtime/openhands.py`, `runtime/deepagents.py`.
- **ClaudeAgentExecutor** (`worker/claude_executor.py`): Implements AgentExecutor interface. Handles task execution, context management, and error handling.
- **MessageConverter** (`worker/converter.py`): Bidirectional conversion between Claude Agent SDK messages and A2A protocol format.

### Two Agent Patterns

1. **Config-only** via `harness.agent()`: Default `client.run()` behavior
2. **Custom loop** via `@harness.agentloop()`: Full control over execution, receives `(prompt, ctx, client)` args

## Code Style

- Line length: 100 chars
- Python 3.11+ features allowed
- Strict mypy enabled
- Ruff rules: E, F, I (isort), UP (pyupgrade), B (bugbear)

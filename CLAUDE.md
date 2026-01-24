# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastHarness wraps the Claude Agent SDK and exposes agents as A2A (Agent-to-Agent) protocol compliant services. It bridges Claude SDK with Google's A2A protocol via the fasta2a library.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest
uv run pytest tests/test_basic.py::TestSkill  # Single test class
uv run pytest -k "test_agent"                  # Pattern match

# Type check
uv run mypy src/

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
    ├── FastA2A (from fasta2a)
    │   └── Exposes A2A endpoints: /.well-known/agent-card.json, JSON-RPC /
    │
    └── ClaudeWorker (worker/claude_worker.py)
        └── Executes tasks using HarnessClient → Claude SDK
```

**Flow**: A2A request → FastA2A → Broker → ClaudeWorker → HarnessClient → Claude SDK → MessageConverter → A2A response

### Key Components

- **FastHarness** (`app.py`): Main entry point. Registers agents, creates FastA2A app with lifespan management.
- **HarnessClient** (`client.py`): Wraps ClaudeSDKClient. Provides `run()` for full execution and `stream()` for event-based iteration.
- **ClaudeWorker** (`worker/claude_worker.py`): Implements fasta2a Worker. Handles task execution, context management, and error handling.
- **MessageConverter** (`worker/converter.py`): Bidirectional conversion between Claude SDK messages and A2A protocol format.

### Two Agent Patterns

1. **Config-only** via `harness.agent()`: Default `client.run()` behavior
2. **Custom loop** via `@harness.agentloop()`: Full control over execution, receives `(prompt, ctx, client)` args

## Code Style

- Line length: 100 chars
- Python 3.11+ features allowed
- Strict mypy enabled
- Ruff rules: E, F, I (isort), UP (pyupgrade), B (bugbear)

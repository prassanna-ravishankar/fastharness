"""Live integration tests — require a real ANTHROPIC_API_KEY.

Start a real uvicorn server and make actual Claude API calls.
Deselected by default. Run with: uv run pytest -m integration
"""

import asyncio

import httpx
import pytest
import uvicorn
from dotenv import load_dotenv

from fastharness import AgentContext, FastHarness, HarnessClient, Skill

load_dotenv()

PORT_AGENT = 8780
PORT_AGENTLOOP = 8781


async def send(
    client: httpx.AsyncClient,
    port: int,
    text: str,
    msg_id: int,
    context_id: str,
    skill_id: str = "chat",
) -> str | None:
    """Send a JSON-RPC message/send and return the reply text."""
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": f"msg-{msg_id}",
                "role": "user",
                "contextId": context_id,
                "parts": [{"kind": "text", "text": text}],
            },
            "metadata": {"skill_id": skill_id},
        },
        "id": msg_id,
    }
    response = await client.post(f"http://localhost:{port}/", json=payload, timeout=60.0)
    result = response.json()
    if "result" not in result:
        return None
    message_data = result["result"]
    if "message" in message_data:
        message_data = message_data["message"]
    parts = message_data.get("parts", [])
    return " ".join(p.get("text", "") for p in parts if p.get("kind") == "text") or None


async def start_server(app, port: int) -> tuple[asyncio.Task, uvicorn.Server]:
    """Start a uvicorn server on the given port and wait for it to be ready."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(2)
    return task, server


@pytest.mark.integration
async def test_agent_multiturn():
    """harness.agent() config-only path: multi-turn memory via client pool."""
    harness = FastHarness(
        name="live-agent",
        description="Live integration test",
        url=f"http://localhost:{PORT_AGENT}",
    )
    harness.agent(
        name="chatbot",
        description="Conversational assistant",
        skills=[Skill(id="chat", name="Chat", description="Chat")],
        system_prompt="You are a helpful assistant. Keep responses brief. Remember what the user tells you.",
    )

    task, server = await start_server(harness.app, PORT_AGENT)
    try:
        async with httpx.AsyncClient() as client:
            ctx = "agent-ctx-123"

            reply1 = await send(client, PORT_AGENT, "My name is Alice. Remember it!", 1, ctx)
            assert reply1 is not None, "No reply on turn 1"

            reply2 = await send(client, PORT_AGENT, "What's my name?", 2, ctx)
            assert reply2 is not None, "No reply on turn 2"
            assert "alice" in reply2.lower(), f"Agent forgot name: {reply2}"

            reply3 = await send(client, PORT_AGENT, "I'm thinking of the number 42.", 3, ctx)
            assert reply3 is not None, "No reply on turn 3"

            reply4 = await send(client, PORT_AGENT, "What number was I thinking of?", 4, ctx)
            assert reply4 is not None, "No reply on turn 4"
            assert "42" in reply4, f"Agent forgot number: {reply4}"

            reply5 = await send(client, PORT_AGENT, "Tell me my name and the number.", 5, ctx)
            assert reply5 is not None, "No reply on turn 5"
            assert "alice" in reply5.lower(), f"Agent forgot name on turn 5: {reply5}"
            assert "42" in reply5, f"Agent forgot number on turn 5: {reply5}"
    finally:
        server.should_exit = True
        await task


@pytest.mark.integration
async def test_agentloop_custom_execution():
    """@harness.agentloop() custom loop path: verifies agent.func is invoked."""
    harness = FastHarness(
        name="live-agentloop",
        description="Live agentloop integration test",
        url=f"http://localhost:{PORT_AGENTLOOP}",
    )

    loop_invocations: list[str] = []

    @harness.agentloop(
        name="looper",
        description="Custom loop agent",
        skills=[Skill(id="loop", name="Loop", description="Custom loop")],
        system_prompt="You are a helpful assistant. Keep responses to one sentence.",
    )
    async def looper(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
        loop_invocations.append(prompt)
        result = await client.run(prompt)
        return f"[custom] {result}"

    task, server = await start_server(harness.app, PORT_AGENTLOOP)
    try:
        async with httpx.AsyncClient() as client:
            reply = await send(client, PORT_AGENTLOOP, "Say hello.", 1, "loop-ctx-123", skill_id="loop")
            assert reply is not None, "No reply from agentloop agent"
            assert "[custom]" in reply, f"Custom loop wrapper not applied: {reply}"
            assert loop_invocations, "Agent function was never called"
    finally:
        server.should_exit = True
        await task

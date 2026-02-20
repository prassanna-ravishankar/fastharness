"""Test multi-turn conversation with client pooling.

This script starts a test server and sends multiple messages in the same context
to verify that conversation history is maintained across requests.
"""

import asyncio
import json

import httpx
import pytest
from dotenv import load_dotenv

from fastharness import FastHarness, Skill

load_dotenv()


async def send_message(client: httpx.AsyncClient, context_id: str, text: str, msg_id: int):
    """Send a message to the agent."""
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
            "metadata": {"skill_id": "chat"},
        },
        "id": msg_id,
    }

    print(f"\n📤 Sending: {text}")
    response = await client.post("http://localhost:8766/", json=payload, timeout=60.0)
    result = response.json()

    if "result" in result:
        # Handle both nested and flat result structures
        message_data = result["result"]
        if "message" in message_data:
            message_data = message_data["message"]

        reply_parts = message_data.get("parts", [])
        reply_text = " ".join(p.get("text", "") for p in reply_parts if p.get("kind") == "text")
        print(f"📥 Reply: {reply_text}")
        return reply_text
    else:
        print(f"❌ Unexpected response: {json.dumps(result, indent=2)}")
        return None


@pytest.mark.integration
async def test_multiturn():
    """Test multi-turn conversation."""
    print("=" * 70)
    print("Multi-Turn Conversation Test")
    print("=" * 70)

    # Create harness
    harness = FastHarness(
        name="multiturn-test",
        description="Testing multi-turn conversations",
        url="http://localhost:8766",
    )

    harness.agent(
        name="chatbot",
        description="A conversational assistant",
        skills=[
            Skill(
                id="chat",
                name="Chat",
                description="Have conversations and remember context",
            )
        ],
        system_prompt=(
            "You are a helpful assistant. Keep your responses brief (1-2 sentences). "
            "Remember what the user tells you in the conversation."
        ),
    )

    # Start server in background
    import uvicorn

    config = uvicorn.Config(
        harness.app, host="127.0.0.1", port=8766, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(2)

    try:
        async with httpx.AsyncClient() as client:
            context_id = "test-context-123"

            print("\n🧪 Test 1: Remember a name")
            print("-" * 70)
            reply1 = await send_message(
                client, context_id, "My name is Alice. Remember it!", 1
            )
            assert reply1 is not None, "No reply from agent"

            print("\n🧪 Test 2: Recall the name")
            print("-" * 70)
            reply2 = await send_message(client, context_id, "What's my name?", 2)
            assert reply2 is not None, "No reply from agent"
            assert "alice" in reply2.lower(), f"Agent didn't remember name! Reply: {reply2}"

            print("\n🧪 Test 3: Remember a number")
            print("-" * 70)
            reply3 = await send_message(
                client, context_id, "I'm thinking of the number 42. Don't forget it!", 3
            )
            assert reply3 is not None, "No reply from agent"

            print("\n🧪 Test 4: Recall the number")
            print("-" * 70)
            reply4 = await send_message(client, context_id, "What number was I thinking of?", 4)
            assert reply4 is not None, "No reply from agent"
            assert "42" in reply4, f"Agent didn't remember number! Reply: {reply4}"

            print("\n🧪 Test 5: Recall both name and number")
            print("-" * 70)
            reply5 = await send_message(
                client, context_id, "Tell me my name and the number I was thinking of.", 5
            )
            assert reply5 is not None, "No reply from agent"
            assert "alice" in reply5.lower(), f"Agent forgot name! Reply: {reply5}"
            assert "42" in reply5, f"Agent forgot number! Reply: {reply5}"

            print("\n" + "=" * 70)
            print("✅ All tests passed! Multi-turn conversation works!")
            print("=" * 70)
            print("\nClient pool successfully maintained conversation history across 5 turns.")
            print("The agent remembered:")
            print("  • The user's name (Alice)")
            print("  • The number (42)")
            print("  • Both pieces of information together")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        # Shutdown server
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    try:
        asyncio.run(test_multiturn())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        raise

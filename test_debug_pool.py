"""Debug test to verify client pooling is working."""

import asyncio
import json

import httpx
from dotenv import load_dotenv

from fastharness import FastHarness, Skill

load_dotenv()


async def test_pool_debug():
    """Test with detailed logging."""
    # Enable debug logging
    import logging

    class ExtraFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            if hasattr(record, "context_id"):
                msg += f" [context_id={record.context_id}]"
            if hasattr(record, "params_context_id"):
                msg += f" [params_context_id={record.params_context_id}]"
            if hasattr(record, "is_new"):
                msg += f" [is_new={record.is_new}]"
            if hasattr(record, "pool_size"):
                msg += f" [pool_size={record.pool_size}]"
            return msg

    handler = logging.StreamHandler()
    handler.setFormatter(
        ExtraFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)

    harness = FastHarness(
        name="pool-debug-test",
        description="Debug client pooling",
        url="http://localhost:8767",
    )

    harness.agent(
        name="chatbot",
        description="Test bot",
        skills=[Skill(id="chat", name="Chat", description="Chat")],
        system_prompt="You are a test bot. Keep responses very brief.",
    )

    import uvicorn

    config = uvicorn.Config(
        harness.app, host="127.0.0.1", port=8767, log_level="debug", access_log=True
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(2)

    try:
        async with httpx.AsyncClient() as client:
            context_id = "debug-ctx"

            # Message 1
            print("\n" + "=" * 70)
            print("MESSAGE 1")
            print("=" * 70)
            payload1 = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello, remember the word BANANA"}],
                    },
                    "metadata": {"skill_id": "chat", "conversation_id": context_id},
                },
                "id": 1,
            }
            response1 = await client.post("http://localhost:8767/", json=payload1, timeout=60.0)
            print(f"Response 1: {json.dumps(response1.json(), indent=2)}")

            await asyncio.sleep(1)

            # Message 2
            print("\n" + "=" * 70)
            print("MESSAGE 2 (should reuse client)")
            print("=" * 70)
            payload2 = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-2",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "What word did I tell you to remember?"}],
                    },
                    "metadata": {"skill_id": "chat", "conversation_id": context_id},
                },
                "id": 2,
            }
            response2 = await client.post("http://localhost:8767/", json=payload2, timeout=60.0)
            print(f"Response 2: {json.dumps(response2.json(), indent=2)}")

    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(test_pool_debug())

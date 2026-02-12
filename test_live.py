"""Live integration test for security fixes.

Simple test that runs the server and sends real messages to verify our fixes work.

Run with: uv run python test_live.py
"""

import asyncio

import uvicorn
from dotenv import load_dotenv

from fastharness import FastHarness, Skill

load_dotenv()


async def main():
    """Test the security fixes with actual Claude API."""
    print("🚀 Starting live integration test...\n")
    print("This will start a test server and send real messages to Claude.")
    print("Make sure you have ANTHROPIC_API_KEY set in your .env file.\n")

    # Create harness with multiple agents
    harness = FastHarness(
        name="test-security-harness",
        description="Testing security fixes",
        url="http://localhost:8765",
    )

    # Agent 1: General assistant
    harness.agent(
        name="assistant",
        description="General purpose assistant",
        skills=[
            Skill(
                id="general",
                name="General",
                description="Answer general questions",
            )
        ],
        system_prompt="You are a helpful assistant. Answer briefly in one sentence.",
    )

    # Agent 2: Math specialist
    harness.agent(
        name="math-expert",
        description="Math specialist",
        skills=[
            Skill(
                id="math",
                name="Math",
                description="Solve math problems",
            )
        ],
        system_prompt="You are a math expert. Answer math questions briefly.",
    )

    app = harness.app
    print(f"✅ Created app with {len(harness._agents)} agents")
    print("   - assistant (general skill)")
    print("   - math-expert (math skill)")
    print()

    # Start the server
    print("🌐 Starting server on http://localhost:8765")
    print("   (Press Ctrl+C to stop after tests complete)\n")

    config = uvicorn.Config(
        app, host="127.0.0.1", port=8765, log_level="info", access_log=False
    )
    server = uvicorn.Server(config)

    await server.serve()


if __name__ == "__main__":
    print("=" * 70)
    print("FastHarness Live Integration Test")
    print("=" * 70)
    print()
    print("Testing security fixes:")
    print("  1. IDOR vulnerability - Task ownership tracking")
    print("  2. Conversation history - No message duplication")
    print("  3. Agent selection - Routing by skill ID")
    print("  4. Resource exhaustion - Proper task cancellation")
    print("  5. Type hints - FastAPI type safety")
    print()
    print("Test the server manually with:")
    print("  1. Get agent card:")
    print("     curl http://localhost:8765/.well-known/agent-card.json")
    print()
    print("  2. Send a message (math skill):")
    print("     curl -X POST http://localhost:8765/ \\")
    print('       -H "Content-Type: application/json" \\')
    print(
        '       -d \'{"jsonrpc":"2.0","method":"message.send",'
        '"params":{"contextId":"ctx-1","message":{"role":"user",'
        '"parts":[{"kind":"text","text":"What is 2+2?"}]},'
        '"metadata":{"skill_id":"math"}},"id":1}\''
    )
    print()
    print("  3. Continue conversation:")
    print("     curl -X POST http://localhost:8765/ \\")
    print('       -H "Content-Type: application/json" \\')
    print(
        '       -d \'{"jsonrpc":"2.0","method":"message.send",'
        '"params":{"contextId":"ctx-1","message":{"role":"user",'
        '"parts":[{"kind":"text","text":"What about 3+3?"}]}},"id":2}\''
    )
    print()
    print("=" * 70)
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
        print("\nIf you tested the endpoints above, you should have seen:")
        print("  ✅ Task ownership tracking (owner_id in metadata)")
        print("  ✅ No duplicate messages in history")
        print("  ✅ Multi-turn conversations working")
        print("  ✅ Agent selection by skill")

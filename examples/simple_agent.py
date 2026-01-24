"""Simple agent example.

Run with: uvicorn examples.simple_agent:app --port 8000
Test: curl http://localhost:8000/.well-known/agent-card.json
"""

from fastharness import AgentContext, FastHarness, HarnessClient, Skill

harness = FastHarness(
    name="demo-agent",
    description="Demo FastHarness agent",
    version="1.0.0",
)

# Simple agent - config only, no function needed
helper = harness.agent(
    name="helper",
    description="A helpful assistant that answers questions",
    skills=[
        Skill(
            id="help",
            name="Help",
            description="Answer general questions and provide assistance",
        )
    ],
    system_prompt="You are a helpful assistant. Be concise and direct.",
    tools=["Read", "Grep", "Glob"],
)


# Custom agent loop - with function for control
@harness.agentloop(
    name="researcher",
    description="Research assistant with multi-turn capability",
    skills=[
        Skill(
            id="research",
            name="Research",
            description="Conduct deep research on topics",
            tags=["research", "analysis"],
        )
    ],
    system_prompt="You are a thorough researcher. Investigate topics deeply.",
    tools=["Read", "WebSearch", "WebFetch"],
)
async def researcher(prompt: str, ctx: AgentContext, client: HarnessClient) -> str:
    """Multi-turn research: keep going until confident."""
    result = await client.run(prompt)

    # Continue if more research needed
    iterations = 0
    while "need more information" in result.lower() and iterations < 3:
        result = await client.run("Continue researching, go deeper")
        iterations += 1

    return result


# Export the app
app = harness.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Integration tests for FastHarness A2A endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from fastharness import FastHarness, Skill


@asynccontextmanager
async def app_client(harness: FastHarness) -> AsyncIterator[AsyncClient]:
    """Create an async client with the app's lifespan properly started."""
    app = harness.app
    # Run the lifespan in a task group
    async with anyio.create_task_group() as tg:
        # Enter the lifespan context
        lifespan_cm = app.router.lifespan_context(app)
        await lifespan_cm.__aenter__()
        try:
            transport = ASGITransport(app=app)  # type: ignore[arg-type]
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                yield client
        finally:
            await lifespan_cm.__aexit__(None, None, None)
            tg.cancel_scope.cancel()


@pytest.fixture
def test_harness() -> FastHarness:
    """Create a test harness with a simple agent."""
    harness = FastHarness(
        name="test-agent",
        description="Test A2A agent",
        version="1.0.0",
        url="http://testserver",
    )
    harness.agent(
        name="helper",
        description="Test helper agent",
        skills=[
            Skill(
                id="help",
                name="Help",
                description="Provide help and assistance",
            )
        ],
        system_prompt="You are a test helper.",
        tools=["Read"],
    )
    return harness


@pytest.mark.asyncio
async def test_agent_card_endpoint(test_harness: FastHarness) -> None:
    """Test that the agent card endpoint returns valid A2A agent card."""
    async with app_client(test_harness) as client:
        response = await client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    data = response.json()

    # Verify agent card structure
    assert data["name"] == "test-agent"
    assert data["description"] == "Test A2A agent"
    assert data["version"] == "1.0.0"
    assert data["url"] == "http://testserver"
    assert data["protocolVersion"] == "0.3.0"

    # Verify skills
    assert len(data["skills"]) == 1
    skill = data["skills"][0]
    assert skill["id"] == "help"
    assert skill["name"] == "Help"
    assert skill["description"] == "Provide help and assistance"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Native A2A SDK doesn't support HEAD requests for agent card endpoint")
async def test_agent_card_head_request(test_harness: FastHarness) -> None:
    """Test that HEAD request to agent card works."""
    async with app_client(test_harness) as client:
        response = await client.head("/.well-known/agent-card.json")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_docs_endpoint(test_harness: FastHarness) -> None:
    """Test that the docs endpoint returns HTML."""
    async with app_client(test_harness) as client:
        response = await client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_multiple_agents_skills_aggregated() -> None:
    """Test that skills from multiple agents are aggregated in the card."""
    harness = FastHarness(
        name="multi-agent",
        description="Multi-agent service",
        version="1.0.0",
        url="http://testserver",
    )

    harness.agent(
        name="helper",
        description="Helper agent",
        skills=[
            Skill(
                id="help",
                name="Help",
                description="Provide help",
            )
        ],
    )

    harness.agent(
        name="researcher",
        description="Research agent",
        skills=[
            Skill(
                id="research",
                name="Research",
                description="Conduct research",
            ),
            Skill(
                id="analyze",
                name="Analyze",
                description="Analyze data",
            ),
        ],
    )

    async with app_client(harness) as client:
        response = await client.get("/.well-known/agent-card.json")

    data = response.json()

    # Should have 3 skills total (1 from helper + 2 from researcher)
    assert len(data["skills"]) == 3
    skill_ids = [s["id"] for s in data["skills"]]
    assert "help" in skill_ids
    assert "research" in skill_ids
    assert "analyze" in skill_ids

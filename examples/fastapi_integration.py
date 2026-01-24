"""FastAPI integration example.

Shows how to mount FastHarness on an existing FastAPI app.

Run with: uvicorn examples.fastapi_integration:app --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastharness import FastHarness, Skill

# Create FastHarness first (needed for lifespan)
harness = FastHarness(
    name="my-agents",
    description="AI agents for my service",
    version="1.0.0",
    url="http://localhost:8000/agents",  # Note: mounted path
)

harness.agent(
    name="assistant",
    description="General assistant for tasks",
    skills=[
        Skill(
            id="assist",
            name="Assist",
            description="Help with general tasks and questions",
        )
    ],
    system_prompt="You are a helpful assistant.",
    tools=["Read", "Grep"],
)


# Lifespan that starts the harness
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start FastHarness worker and broker."""
    async with harness.lifespan_context():
        yield


# Your existing FastAPI app with lifespan
app = FastAPI(
    title="My Service",
    version="1.0.0",
    description="My service with AI agents",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Your custom routes
@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/custom")
def custom_endpoint() -> dict[str, str]:
    """Custom endpoint."""
    return {"message": "This is my custom endpoint"}


# Mount at /agents - coexists with your routes
app.mount("/agents", harness.app)

# Endpoints:
# GET  /health                             -> Your health check
# GET  /custom                             -> Your custom route
# GET  /agents/.well-known/agent-card.json -> A2A Agent Card
# POST /agents/                            -> A2A JSON-RPC endpoint
# GET  /agents/docs                        -> A2A documentation

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

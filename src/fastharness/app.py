"""FastHarness - Wrap Claude Agent SDK and expose agents as A2A services."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fasta2a import Broker, FastA2A, Storage
from fasta2a.broker import InMemoryBroker
from fasta2a.schema import Skill as A2ASkill
from fasta2a.storage import InMemoryStorage

from fastharness.client import HarnessClient
from fastharness.core.agent import Agent, AgentConfig
from fastharness.core.context import AgentContext
from fastharness.core.skill import Skill
from fastharness.logging import get_logger
from fastharness.worker.claude_worker import AgentRegistry, ClaudeWorker

logger = get_logger("app")

# Type alias for agent function signature
AgentFunc = Callable[[str, AgentContext, HarnessClient], Awaitable[Any]]


class FastHarness:
    """Main class for creating A2A-compliant Claude agents.

    FastHarness wraps the Claude Agent SDK and exposes agents as A2A services
    using FastAPI and fasta2a.

    Example:
        ```python
        from fastharness import FastHarness, Skill

        harness = FastHarness(name="my-service")

        # Simple agent (config-only)
        harness.agent(
            name="assistant",
            description="A helpful assistant",
            skills=[Skill(id="help", name="Help", description="Answer questions")],
        )

        # Or with custom loop
        @harness.agentloop(name="researcher", ...)
        async def researcher(prompt, ctx, client):
            return await client.run(prompt)

        app = harness.app
        ```
    """

    def __init__(
        self,
        name: str = "fastharness-agent",
        description: str = "Claude-powered A2A agent",
        version: str = "1.0.0",
        url: str = "http://localhost:8000",
        storage: Storage[Any] | None = None,
        broker: Broker | None = None,
        tracing: bool = False,
    ):
        """Initialize FastHarness.

        Args:
            name: Name for the A2A agent card.
            description: Description for the A2A agent card.
            version: Version for the A2A agent card.
            url: URL where the agent is hosted.
            storage: Storage implementation (defaults to InMemoryStorage).
            broker: Broker implementation (defaults to InMemoryBroker).
            tracing: Enable OpenTelemetry tracing for all agents (default False).
        """
        self.name = name
        self.description = description
        self.version = version
        self.url = url
        self.tracing = tracing

        self._storage = storage or InMemoryStorage()
        self._broker = broker or InMemoryBroker()
        self._agents: dict[str, Agent] = {}
        self._app: FastA2A | None = None

    def _convert_skills(self, skills: list[Skill]) -> list[A2ASkill]:
        """Convert FastHarness Skills to A2A Skills."""
        return [
            A2ASkill(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=s.tags,
                input_modes=s.input_modes,
                output_modes=s.output_modes,
            )
            for s in skills
        ]

    def _collect_all_skills(self) -> list[A2ASkill]:
        """Collect all skills from all registered agents."""
        all_skills: list[A2ASkill] = []
        for agent in self._agents.values():
            all_skills.extend(self._convert_skills(agent.config.skills))
        return all_skills

    def agent(
        self,
        name: str,
        description: str,
        skills: list[Skill],
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int | None = None,
        model: str = "claude-sonnet-4-20250514",
        mcp_servers: dict[str, Any] | None = None,
        setting_sources: list[str] | None = None,
        output_format: dict[str, Any] | None = None,
        tracing: bool | None = None,
    ) -> Agent:
        """Register a simple agent (config-only, no custom loop).

        The agent will use the default client.run() behavior.

        Args:
            name: Unique name for the agent.
            description: Human-readable description.
            skills: List of A2A skills this agent provides.
            system_prompt: System prompt for Claude.
            tools: List of allowed tool names (e.g., ["Read", "Grep", "Glob"]).
            max_turns: Maximum number of turns.
            model: Claude model to use.
            mcp_servers: MCP servers to connect (dict of name -> config).
            setting_sources: Filesystem setting sources to load (["project"] loads CLAUDE.md).
            output_format: JSON schema for structured output
                (e.g., {"type": "json_schema", "schema": {...}}).
            tracing: Enable OTel tracing for this agent (None inherits from harness).

        Returns:
            The registered Agent.
        """
        config = AgentConfig(
            name=name,
            description=description,
            skills=skills,
            system_prompt=system_prompt,
            tools=tools or [],
            max_turns=max_turns,
            model=model,
            mcp_servers=mcp_servers or {},
            setting_sources=setting_sources if setting_sources is not None else ["project"],
            output_format=output_format,
            tracing=tracing if tracing is not None else self.tracing,
        )
        agent = Agent(config=config, func=None)
        self._agents[name] = agent
        self._app = None  # Invalidate cached app
        logger.info(
            "Registered agent",
            extra={"agent_name": name, "skill_count": len(skills)},
        )
        return agent

    def agentloop(
        self,
        name: str,
        description: str,
        skills: list[Skill],
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int | None = None,
        model: str = "claude-sonnet-4-20250514",
        mcp_servers: dict[str, Any] | None = None,
        setting_sources: list[str] | None = None,
        output_format: dict[str, Any] | None = None,
        tracing: bool | None = None,
    ) -> Callable[[AgentFunc], Agent]:
        """Decorator to register an agent with custom loop logic.

        The decorated function controls the execution loop.

        Args:
            name: Unique name for the agent.
            description: Human-readable description.
            skills: List of A2A skills this agent provides.
            system_prompt: System prompt for Claude.
            tools: List of allowed tool names.
            max_turns: Maximum number of turns.
            model: Claude model to use.
            mcp_servers: MCP servers to connect (dict of name -> config).
            setting_sources: Filesystem setting sources to load (["project"] loads CLAUDE.md).
            output_format: JSON schema for structured output
                (e.g., {"type": "json_schema", "schema": {...}}).
            tracing: Enable OTel tracing for this agent (None inherits from harness).

        Returns:
            Decorator that registers the agent function.

        Example:
            ```python
            @harness.agentloop(name="researcher", ...)
            async def researcher(prompt: str, ctx: AgentContext, client: HarnessClient):
                result = await client.run(prompt)
                while "need_more" in result:
                    result = await client.run("Continue")
                return result
            ```
        """

        def decorator(func: AgentFunc) -> Agent:
            config = AgentConfig(
                name=name,
                description=description,
                skills=skills,
                system_prompt=system_prompt,
                tools=tools or [],
                max_turns=max_turns,
                model=model,
                mcp_servers=mcp_servers or {},
                setting_sources=setting_sources if setting_sources is not None else ["project"],
                output_format=output_format,
                tracing=tracing if tracing is not None else self.tracing,
            )
            agent = Agent(config=config, func=func)
            self._agents[name] = agent
            self._app = None  # Invalidate cached app
            logger.info(
                "Registered agent with custom loop",
                extra={"agent_name": name, "skill_count": len(skills)},
            )
            return agent

        return decorator

    def _create_app(self) -> FastA2A:
        """Create the FastA2A application."""
        registry = AgentRegistry(agents=self._agents)

        # Create worker
        worker = ClaudeWorker(
            broker=self._broker,
            storage=self._storage,
            agent_registry=registry,
        )

        @asynccontextmanager
        async def lifespan(app: FastA2A) -> AsyncIterator[None]:
            async with self._broker:
                async with worker.run():
                    async with app.task_manager:
                        yield

        return FastA2A(
            storage=self._storage,
            broker=self._broker,
            name=self.name,
            description=self.description,
            version=self.version,
            url=self.url,
            skills=self._collect_all_skills(),
            lifespan=lifespan,
        )

    @asynccontextmanager
    async def lifespan_context(self) -> AsyncIterator[None]:
        """Context manager to start the harness components.

        Use this when mounting FastHarness on another FastAPI app.
        The parent app's lifespan should wrap this context manager.

        Example:
            ```python
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                async with harness.lifespan_context():
                    yield

            app = FastAPI(lifespan=lifespan)
            app.mount("/agents", harness.app)
            ```
        """
        # Ensure app is created
        fasta2a_app = self.app

        registry = AgentRegistry(agents=self._agents)
        worker = ClaudeWorker(
            broker=self._broker,
            storage=self._storage,
            agent_registry=registry,
        )

        async with self._broker:
            async with worker.run():
                async with fasta2a_app.task_manager:
                    yield

    @property
    def app(self) -> FastA2A:
        """Return FastAPI-compatible app with A2A endpoints.

        The app can be:
        - Run directly: `uvicorn mymodule:harness.app`
        - Mounted on another FastAPI app: `fastapi_app.mount("/agents", harness.app)`

        Returns:
            FastA2A application (Starlette-based, FastAPI-compatible).
        """
        if self._app is None:
            self._app = self._create_app()
        return self._app

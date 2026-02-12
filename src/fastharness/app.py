"""FastHarness - Wrap Claude Agent SDK and expose agents as A2A services."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_store import TaskStore
from a2a.types import AgentCapabilities, AgentCard
from a2a.types import AgentSkill as A2ASkill

from fastharness.client import HarnessClient
from fastharness.core.agent import Agent, AgentConfig
from fastharness.core.context import AgentContext
from fastharness.core.skill import Skill
from fastharness.logging import get_logger
from fastharness.worker.claude_executor import AgentRegistry, ClaudeAgentExecutor

logger = get_logger("app")

# Type alias for agent function signature
AgentFunc = Callable[[str, AgentContext, HarnessClient], Awaitable[Any]]


class FastHarness:
    """Main class for creating A2A-compliant Claude agents.

    FastHarness wraps the Claude Agent SDK and exposes agents as A2A services
    using FastAPI and the native A2A Python SDK.

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
        task_store: TaskStore | None = None,
    ):
        """Initialize FastHarness.

        Args:
            name: Name for the A2A agent card.
            description: Description for the A2A agent card.
            version: Version for the A2A agent card.
            url: URL where the agent is hosted.
            task_store: TaskStore implementation (defaults to InMemoryTaskStore).
        """
        self.name = name
        self.description = description
        self.version = version
        self.url = url
        self._task_store = task_store or InMemoryTaskStore()
        self._agents: dict[str, Agent] = {}
        self._app: FastAPI | None = None

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

    def _register_agent(
        self,
        name: str,
        description: str,
        skills: list[Skill],
        func: AgentFunc | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int | None = None,
        model: str = "claude-sonnet-4-20250514",
        mcp_servers: dict[str, Any] | None = None,
        setting_sources: list[str] | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> Agent:
        """Build AgentConfig, create Agent, and register it.

        Shared implementation for both agent() and agentloop().
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
        )
        agent = Agent(config=config, func=func)
        self._agents[name] = agent
        self._app = None  # Invalidate cached app
        logger.info(
            "Registered agent",
            extra={
                "agent_name": name,
                "skill_count": len(skills),
                "has_custom_loop": func is not None,
            },
        )
        return agent

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

        Returns:
            The registered Agent.
        """
        return self._register_agent(
            name=name,
            description=description,
            skills=skills,
            system_prompt=system_prompt,
            tools=tools,
            max_turns=max_turns,
            model=model,
            mcp_servers=mcp_servers,
            setting_sources=setting_sources,
            output_format=output_format,
        )

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
            return self._register_agent(
                name=name,
                description=description,
                skills=skills,
                func=func,
                system_prompt=system_prompt,
                tools=tools,
                max_turns=max_turns,
                model=model,
                mcp_servers=mcp_servers,
                setting_sources=setting_sources,
                output_format=output_format,
            )

        return decorator

    def _create_app(self) -> "FastAPI":
        """Create the FastAPI application with native A2A SDK."""
        registry = AgentRegistry(agents=self._agents)

        # Create AgentCard
        agent_card = AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            url=self.url,
            skills=self._collect_all_skills(),
            capabilities=AgentCapabilities(),  # Default capabilities
            default_input_modes=["text"],  # Default to text input
            default_output_modes=["text"],  # Default to text output
        )

        # Create ClaudeAgentExecutor
        executor = ClaudeAgentExecutor(
            agent_registry=registry,
            task_store=self._task_store,
        )

        # Create DefaultRequestHandler (handles all JSON-RPC methods)
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=self._task_store,
            queue_manager=None,  # Use SDK default (InMemoryQueueManager)
        )

        # Build FastAPI app with A2A endpoints
        a2a_app = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=handler,
        )

        return a2a_app.build()

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
        _ = self.app

        # Native SDK manages lifecycle internally
        # Task store doesn't require explicit lifecycle management
        yield

    @property
    def app(self) -> "FastAPI":
        """Return FastAPI-compatible app with A2A endpoints.

        The app can be:
        - Run directly: `uvicorn mymodule:harness.app`
        - Mounted on another FastAPI app: `fastapi_app.mount("/agents", harness.app)`

        Returns:
            FastAPI application with native A2A SDK integration.
        """
        if self._app is None:
            self._app = self._create_app()
        return self._app

"""ClaudeAgentExecutor - Claude SDK integration with native A2A AgentExecutor."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_store import TaskStore
from a2a.types import Artifact, Message, Role, Task, TaskState, TaskStatus

from fastharness.client import HarnessClient
from fastharness.core.context import AgentContext
from fastharness.core.context import Message as ContextMessage
from fastharness.logging import get_logger
from fastharness.worker.client_pool import ClientPool
from fastharness.worker.converter import MessageConverter

if TYPE_CHECKING:
    from fastharness.core.agent import Agent

logger = get_logger("executor")


def _ensure_history(task: Task) -> list[Message]:
    """Ensure task.history is a list, initializing it if None."""
    if task.history is None:
        task.history = []
    return task.history


def _get_user_id(context: RequestContext) -> str:
    """Extract authenticated user ID from request context."""
    user = context.call_context.user if context.call_context else None
    if user and user.is_authenticated:
        return user.user_name
    return "anonymous"


def _authorize_task_access(task: Task, context: RequestContext) -> bool:
    """Check if requesting user owns this task."""
    if not task.metadata:
        return True  # Legacy tasks without owner

    task_owner = task.metadata.get("owner_id")
    if not task_owner:
        return True  # Legacy tasks

    current_user = _get_user_id(context)
    return task_owner == current_user


def _extract_requested_skill(context: RequestContext) -> str | None:
    """Extract requested skill ID from request context."""
    if context.metadata:
        return context.metadata.get("skill_id") or None
    return None


@dataclass
class AgentRegistry:
    """Registry of agents available to the executor."""

    agents: dict[str, "Agent"]

    def __post_init__(self) -> None:
        """Build skill-to-agent mapping."""
        self._skill_to_agent: dict[str, str] = {}
        for agent_name, agent in self.agents.items():
            for skill in agent.config.skills:
                if skill.id not in self._skill_to_agent:
                    self._skill_to_agent[skill.id] = agent_name

    def get(self, name: str) -> "Agent | None":
        """Get agent by name."""
        return self.agents.get(name)

    def get_by_skill(self, skill_id: str) -> "Agent | None":
        """Get agent that provides the given skill."""
        agent_name = self._skill_to_agent.get(skill_id)
        return self.agents.get(agent_name) if agent_name else None

    def get_default(self) -> "Agent | None":
        """Get the default (first registered) agent."""
        if self.agents:
            return next(iter(self.agents.values()))
        return None

    def resolve(self, skill_id: str | None = None) -> "Agent | None":
        """Resolve an agent by skill ID, falling back to the default agent."""
        if skill_id:
            agent = self.get_by_skill(skill_id)
            if agent:
                return agent
            logger.warning(
                "Requested skill not found, falling back to default agent",
                extra={"skill_id": skill_id},
            )
        return self.get_default()


@dataclass
class ClaudeAgentExecutor(AgentExecutor):
    """AgentExecutor implementation that executes tasks using Claude SDK.

    Bridges the A2A protocol with the Claude Agent SDK.
    """

    agent_registry: AgentRegistry
    task_store: TaskStore

    def __post_init__(self) -> None:
        """Initialize task tracking and client pool."""
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._client_pool = ClientPool(ttl_minutes=15)

    def build_message_history(self, history: list[Message]) -> list[Any]:
        """Convert A2A message history to Claude SDK format."""
        return MessageConverter.a2a_to_claude_messages(history)

    def build_artifacts(self, result: Any) -> list[Artifact]:
        """Convert execution result to A2A artifacts."""
        if isinstance(result, str):
            return [MessageConverter.text_to_artifact(result)]
        if isinstance(result, list):
            return result
        if result is not None:
            return [MessageConverter.text_to_artifact(str(result))]
        return []

    async def _fail_task(
        self,
        task: Task,
        error_content: str,
        event_queue: EventQueue,
    ) -> None:
        """Mark a task as failed with an error message and enqueue the event."""
        error_message = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content=error_content,
        )
        task.status = TaskStatus(state=TaskState.failed, message=error_message)
        _ensure_history(task).append(error_message)
        await self.task_store.save(task)
        await event_queue.enqueue_event(error_message)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a task using the Claude SDK with cancellation tracking."""
        task_id = context.task_id
        if not task_id:
            logger.error("Invalid request context: missing task_id")
            return

        # Create and track execution task for cancellation
        exec_task = asyncio.create_task(
            self._execute_impl(context, event_queue),
            name=f"task-{task_id}",
        )
        self._running_tasks[task_id] = exec_task

        try:
            await exec_task
        except asyncio.CancelledError:
            logger.info("Task execution cancelled", extra={"task_id": task_id})
            raise
        finally:
            self._running_tasks.pop(task_id, None)

    async def _execute_impl(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a task using the Claude SDK.

        1. Extract task info from RequestContext
        2. Get agent from registry
        3. Load task history from TaskStore
        4. Build HarnessClient from agent config
        5. Execute with ClaudeSDKClient (via agent func or default)
        6. Convert results to artifacts
        7. Enqueue response messages via EventQueue
        8. Update task in TaskStore with completed status
        """
        task_id = context.task_id
        context_id = context.context_id
        message = context.message

        if not task_id or not context_id or not message:
            logger.error("Invalid request context: missing task_id, context_id, or message")
            return

        logger.info(
            "Starting task execution",
            extra={"task_id": task_id, "context_id": context_id},
        )

        try:
            # Get current task from context or create new one
            current_task = context.current_task
            if current_task is None:
                current_task = Task(
                    id=task_id,
                    context_id=context_id,
                    kind="task",
                    status=TaskStatus(state=TaskState.working),
                    history=[message],
                    artifacts=[],
                    metadata={"owner_id": _get_user_id(context)},
                )
                await self.task_store.save(current_task)
                context.current_task = current_task
            else:
                # Existing task - A2A SDK already updated history via update_with_message
                # Reload to get fresh history
                current_task = await self.task_store.get(task_id)
                if not current_task:
                    logger.error("Task not found in store", extra={"task_id": task_id})
                    return

                # Check authorization
                if not _authorize_task_access(current_task, context):
                    logger.warning(
                        "Unauthorized task access attempt",
                        extra={"task_id": task_id, "user": _get_user_id(context)},
                    )
                    await self._fail_task(
                        current_task,
                        "Error: Access denied. You do not have permission to access this task.",
                        event_queue,
                    )
                    return

                current_task.status = TaskStatus(state=TaskState.working)
                await self.task_store.save(current_task)
                context.current_task = current_task

            # Resolve agent by requested skill, falling back to default
            requested_skill = _extract_requested_skill(context)
            agent = self.agent_registry.resolve(requested_skill)

            if agent is None:
                logger.error("Task failed: no agents registered", extra={"task_id": task_id})
                await self._fail_task(
                    current_task,
                    "Error: No agents registered. Configure at least one agent.",
                    event_queue,
                )
                return

            logger.info(
                "Using agent for task",
                extra={
                    "task_id": task_id,
                    "agent_name": agent.config.name,
                    "requested_skill": requested_skill,
                },
            )

            # Extract user prompt and build context
            history = current_task.history or []
            prompt = MessageConverter.extract_text_from_parts(message.parts)

            message_history = [
                ContextMessage(
                    role="user" if m.role == Role.user else "assistant",
                    content=MessageConverter.extract_text_from_parts(m.parts),
                )
                for m in history
                if m.role in (Role.user, Role.agent)
            ]

            ctx = AgentContext(
                task_id=task_id,
                context_id=context_id,
                message_history=message_history,
            )

            # Build ClaudeAgentOptions from agent config
            from typing import Literal, cast

            from claude_agent_sdk import ClaudeAgentOptions

            config = agent.config
            options = ClaudeAgentOptions(
                system_prompt=config.system_prompt,
                allowed_tools=config.tools,
                model=config.model,
                max_turns=config.max_turns,
                mcp_servers=config.mcp_servers,
                setting_sources=cast(
                    list[Literal["user", "project", "local"]] | None, config.setting_sources
                ),
                output_format=config.output_format,
                permission_mode="bypassPermissions",
            )

            # Get or create pooled client
            pooled_client, is_new = await self._client_pool.get_or_create(context_id, options)

            # Build HarnessClient with pooled SDK client
            client = HarnessClient(
                system_prompt=config.system_prompt,
                tools=config.tools,
                model=config.model,
                max_turns=config.max_turns,
                mcp_servers=config.mcp_servers,
                setting_sources=config.setting_sources,
                output_format=config.output_format,
                pooled_client=pooled_client,
            )

            # Execute agent
            execution_mode = "custom" if agent.func is not None else "default"
            logger.debug(
                f"Executing {execution_mode} agent run",
                extra={"task_id": task_id, "agent_name": agent.config.name},
            )

            if agent.func is not None:
                result = await agent.func(prompt, ctx, client)
            else:
                result = await client.run(prompt)

            # Convert result to artifacts and response message
            artifacts = self.build_artifacts(result)
            response_message = MessageConverter.claude_to_a2a_message(
                role="assistant",
                content=result if isinstance(result, str) else str(result),
            )

            # Update task with results
            # DO NOT append the user message - it's already in history
            # Only append the assistant's response
            task_history = _ensure_history(current_task)
            task_history.append(response_message)

            current_task.status = TaskStatus(state=TaskState.completed)
            current_task.artifacts = artifacts
            await self.task_store.save(current_task)
            await event_queue.enqueue_event(response_message)

            # Cleanup pooled client on task completion
            await self._client_pool.remove(context_id)

            logger.info(
                "Task completed successfully",
                extra={"task_id": task_id, "artifact_count": len(artifacts)},
            )

        except Exception as e:
            logger.exception(
                "Task failed with exception",
                extra={
                    "task_id": task_id,
                    "context_id": context_id,
                    "error_type": type(e).__name__,
                },
            )

            # Cleanup pooled client on task failure
            await self._client_pool.remove(context_id)

            try:
                if context.current_task:
                    await self._fail_task(
                        context.current_task,
                        f"An error occurred: {type(e).__name__}",
                        event_queue,
                    )
                else:
                    # No task to update, just enqueue the error
                    error_message = MessageConverter.claude_to_a2a_message(
                        role="assistant",
                        content=f"An error occurred: {type(e).__name__}",
                    )
                    await event_queue.enqueue_event(error_message)
            except Exception:
                logger.exception(
                    "Failed to update task to failed state",
                    extra={"task_id": task_id},
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        task_id = context.task_id

        if not task_id:
            logger.error("Invalid cancel request: missing task_id")
            return

        logger.info("Cancelling task", extra={"task_id": task_id})

        # Check authorization
        current_task = context.current_task
        if current_task and not _authorize_task_access(current_task, context):
            logger.warning(
                "Unauthorized cancel attempt",
                extra={"task_id": task_id, "user": _get_user_id(context)},
            )
            # Enqueue error message to inform caller
            error_message = MessageConverter.claude_to_a2a_message(
                role="assistant",
                content="Error: Access denied. You do not have permission to cancel this task.",
            )
            await event_queue.enqueue_event(error_message)
            return

        # Cancel the running asyncio task
        running_task = self._running_tasks.get(task_id)
        if running_task and not running_task.done():
            running_task.cancel()
            try:
                await running_task
            except asyncio.CancelledError:
                pass

        # Update task status
        if not current_task:
            return

        # Cleanup pooled client on task cancellation
        if context.context_id:
            await self._client_pool.remove(context.context_id)

        current_task.status = TaskStatus(state=TaskState.canceled)
        await self.task_store.save(current_task)

        cancel_message = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Task cancelled by user request.",
        )
        await event_queue.enqueue_event(cancel_message)

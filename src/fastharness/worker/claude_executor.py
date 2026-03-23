"""ClaudeAgentExecutor - Claude SDK integration with native A2A AgentExecutor."""

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    Artifact,
    Message,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
)

from fastharness.client import HarnessClient
from fastharness.core.context import AgentContext
from fastharness.core.context import Message as ContextMessage
from fastharness.core.event import DoneEvent, TextEvent, ToolEvent
from fastharness.logging import get_logger
from fastharness.runtime.base import AgentRuntimeFactory
from fastharness.worker import converter
from fastharness.worker.converter import MessageConverter

if TYPE_CHECKING:
    from fastharness.core.agent import Agent

logger = get_logger("executor")


@dataclass(frozen=True)
class HarnessRequestMetadata:
    """Typed view over FastHarness extension keys in RequestContext.metadata.

    Parsed once at the request boundary so downstream code can use
    typed fields instead of repeated dict.get() calls.
    """

    skill_id: str | None = None

    @classmethod
    def from_context(cls, context: RequestContext) -> "HarnessRequestMetadata":
        """Extract FastHarness extension fields from a RequestContext."""
        raw = context.metadata or {}
        return cls(
            skill_id=str(raw["skill_id"]) if raw.get("skill_id") else None,
        )


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
    runtime_factory: AgentRuntimeFactory | None = None

    def __post_init__(self) -> None:
        """Initialize task tracking and resolve runtime factory."""
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        if self.runtime_factory is None:
            from fastharness.runtime.claude import ClaudeRuntimeFactory

            self.runtime_factory = ClaudeRuntimeFactory(ttl_minutes=15)
        assert self.runtime_factory is not None  # guaranteed after __post_init__

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

        # Parse metadata once at the boundary
        meta = HarnessRequestMetadata.from_context(context)

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
            agent = self.agent_registry.resolve(meta.skill_id)

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
                    "requested_skill": meta.skill_id,
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

            # Get or create a runtime scoped to user+context
            config = agent.config
            pool_key = f"{_get_user_id(context)}:{context_id}"
            runtime = await self.runtime_factory.get_or_create(pool_key, config)

            # Build HarnessClient with runtime
            client = HarnessClient(
                system_prompt=config.system_prompt,
                tools=config.tools,
                model=config.model,
                max_turns=config.max_turns,
                mcp_servers=config.mcp_servers,
                setting_sources=config.setting_sources,
                output_format=config.output_format,
                runtime=runtime,
            )

            # Execute agent
            if agent.func is not None:
                # Custom loop — run() only, loop controls execution
                logger.debug(
                    "Executing custom agent run",
                    extra={"task_id": task_id, "agent_name": agent.config.name},
                )
                result = await agent.func(prompt, ctx, client)
                await self._complete_task(
                    current_task, result, task_id, context_id, event_queue
                )
            else:
                # Default agent — stream tokens as A2A artifact updates
                logger.debug(
                    "Executing streaming agent run",
                    extra={"task_id": task_id, "agent_name": agent.config.name},
                )
                await self._stream_task(
                    current_task, client, prompt, task_id, context_id, event_queue
                )

            logger.info(
                "Task completed successfully",
                extra={
                    "task_id": task_id,
                    "artifact_count": len(current_task.artifacts or []),
                },
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

            # Cleanup runtime on task failure
            await self.runtime_factory.remove(f"{_get_user_id(context)}:{context_id}")

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

    async def _complete_task(
        self,
        task: Task,
        result: Any,
        task_id: str,
        context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """Finalize a task with a complete result (used by custom loops)."""
        artifacts = self.build_artifacts(result)
        response_message = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content=result if isinstance(result, str) else str(result),
        )

        task_history = _ensure_history(task)
        task_history.append(response_message)

        task.status = TaskStatus(state=TaskState.completed)
        task.artifacts = artifacts
        await self.task_store.save(task)
        await event_queue.enqueue_event(response_message)

    async def _stream_task(
        self,
        task: Task,
        client: HarnessClient,
        prompt: str,
        task_id: str,
        context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """Stream agent execution, emitting A2A artifact updates as tokens arrive."""

        artifact_id = str(uuid.uuid4())
        chunks: list[str] = []
        chunk_count = 0

        async for event in client.stream(prompt):
            if isinstance(event, ToolEvent):
                logger.debug(
                    "Tool call during stream",
                    extra={"tool_name": event.tool_name, "task_id": task_id},
                )
                continue
            if isinstance(event, TextEvent):
                chunk_count += 1
                chunks.append(event.text)
                # Emit incremental artifact update
                chunk_artifact = Artifact(
                    artifact_id=artifact_id,
                    name="response",
                    parts=[converter._text_part(event.text)],
                )
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        artifact=chunk_artifact,
                        append=chunk_count > 1,
                    )
                )
            elif isinstance(event, DoneEvent):
                final_text = event.final_text or "".join(chunks)
                # Emit last-chunk artifact marker
                final_artifact = Artifact(
                    artifact_id=artifact_id,
                    name="response",
                    parts=[converter._text_part("")],
                )
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        artifact=final_artifact,
                        append=True,
                        last_chunk=True,
                    )
                )

                # Finalize task state
                response_message = MessageConverter.claude_to_a2a_message(
                    role="assistant", content=final_text
                )
                task_history = _ensure_history(task)
                task_history.append(response_message)

                complete_artifact = MessageConverter.text_to_artifact(final_text)
                task.status = TaskStatus(state=TaskState.completed)
                task.artifacts = [complete_artifact]
                await self.task_store.save(task)

                # Emit the Message — required for message/send compatibility
                await event_queue.enqueue_event(response_message)
                break

        # If stream ended without DoneEvent (shouldn't happen, but handle it)
        if task.status.state != TaskState.completed:
            await self._complete_task(
                task, "".join(chunks), task_id, context_id, event_queue
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

        # Cleanup runtime on task cancellation
        if context.context_id:
            await self.runtime_factory.remove(f"{_get_user_id(context)}:{context.context_id}")

        current_task.status = TaskStatus(state=TaskState.canceled)
        await self.task_store.save(current_task)

        cancel_message = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Task cancelled by user request.",
        )
        await event_queue.enqueue_event(cancel_message)

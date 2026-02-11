"""ClaudeAgentExecutor - Claude SDK integration with native A2A AgentExecutor."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_store import TaskStore
from a2a.types import Artifact, Message, Task, TaskState, TaskStatus

from fastharness.client import HarnessClient
from fastharness.core.context import AgentContext
from fastharness.core.context import Message as ContextMessage
from fastharness.logging import get_logger
from fastharness.worker.converter import MessageConverter

if TYPE_CHECKING:
    from fastharness.core.agent import Agent

logger = get_logger("executor")


def _ensure_history(task: Task) -> list[Message]:
    """Ensure task.history is a list, initializing it if None."""
    if task.history is None:
        task.history = []
    return task.history


@dataclass
class AgentRegistry:
    """Registry of agents available to the executor."""

    agents: dict[str, "Agent"]

    def get(self, name: str) -> "Agent | None":
        """Get agent by name."""
        return self.agents.get(name)

    def get_default(self) -> "Agent | None":
        """Get the default (first registered) agent."""
        if self.agents:
            return next(iter(self.agents.values()))
        return None


@dataclass
class ClaudeAgentExecutor(AgentExecutor):
    """AgentExecutor implementation that executes tasks using Claude SDK.

    Bridges the A2A protocol with the Claude Agent SDK.
    """

    agent_registry: AgentRegistry
    task_store: TaskStore

    def __post_init__(self) -> None:
        """Initialize task tracking."""
        self._running_tasks: dict[str, bool] = {}

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

        # Track running task for cancellation
        self._running_tasks[task_id] = True

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
                )
                await self.task_store.save(current_task)
                context.current_task = current_task
            else:
                current_task.status = TaskStatus(state=TaskState.working)
                await self.task_store.save(current_task)

            # Get agent (use default if only one registered)
            agent = self.agent_registry.get_default()
            if agent is None:
                logger.error(
                    "Task failed: no agents registered",
                    extra={"task_id": task_id},
                )
                await self._fail_task(
                    current_task,
                    "Error: No agents registered. Configure at least one agent "
                    "using harness.agent() or @harness.agentloop() before handling tasks.",
                    event_queue,
                )
                return

            logger.debug(
                "Using agent for task",
                extra={"task_id": task_id, "agent_name": agent.config.name},
            )

            # Extract user prompt and build context
            history = current_task.history or []
            prompt = MessageConverter.extract_text_from_parts(message.parts)

            message_history = [
                ContextMessage(
                    role="user" if m.role == "user" else "assistant",
                    content=MessageConverter.extract_text_from_parts(m.parts),
                )
                for m in history
                if m.role in ("user", "assistant")
            ]

            ctx = AgentContext(
                task_id=task_id,
                context_id=context_id,
                message_history=message_history,
            )

            # Build HarnessClient from agent config
            config = agent.config
            client = HarnessClient(
                system_prompt=config.system_prompt,
                tools=config.tools,
                model=config.model,
                max_turns=config.max_turns,
                mcp_servers=config.mcp_servers,
                setting_sources=config.setting_sources,
                output_format=config.output_format,
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
            task_history = _ensure_history(current_task)
            task_history.append(message)
            task_history.append(response_message)

            current_task.status = TaskStatus(state=TaskState.completed)
            current_task.artifacts = artifacts
            await self.task_store.save(current_task)
            await event_queue.enqueue_event(response_message)

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

        finally:
            self._running_tasks.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        task_id = context.task_id

        if not task_id:
            logger.error("Invalid cancel request: missing task_id")
            return

        logger.info("Cancelling task", extra={"task_id": task_id})

        # TODO: Implement actual interruption via client.interrupt()
        # For now, just mark as canceled
        self._running_tasks.pop(task_id, None)

        current_task = context.current_task
        if current_task:
            current_task.status = TaskStatus(state=TaskState.canceled)
            await self.task_store.save(current_task)

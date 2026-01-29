"""ClaudeWorker - Claude SDK integration with fasta2a Worker."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams
from fasta2a.worker import Worker

from fastharness.client import HarnessClient
from fastharness.core.context import AgentContext
from fastharness.core.context import Message as ContextMessage
from fastharness.logging import get_logger
from fastharness.worker.converter import MessageConverter

if TYPE_CHECKING:
    from fastharness.core.agent import Agent

logger = get_logger("worker")


@dataclass
class AgentRegistry:
    """Registry of agents available to the worker."""

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
class ClaudeWorker(Worker[list[dict[str, Any]]]):
    """Worker implementation that executes tasks using Claude SDK.

    Bridges the A2A protocol with the Claude Agent SDK.
    """

    agent_registry: AgentRegistry

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
        elif isinstance(result, list):
            return result
        elif result is not None:
            return [MessageConverter.text_to_artifact(str(result))]
        return []

    async def run_task(self, params: TaskSendParams) -> None:
        """Execute a task using the Claude SDK.

        1. Load context and task from storage
        2. Get agent from registry
        3. Build HarnessClient from agent config
        4. Execute with ClaudeSDKClient (via agent func or default)
        5. Convert messages to A2A format
        6. Update task with artifacts
        """
        task_id = params["id"]
        context_id = params["context_id"]
        message = params["message"]

        logger.info(
            "Starting task execution",
            extra={"task_id": task_id, "context_id": context_id},
        )

        # Track running task for cancellation
        self._running_tasks[task_id] = True

        try:
            # Mark task as working
            await self.storage.update_task(task_id, state="working")

            # Get agent (use default if only one registered)
            agent = self.agent_registry.get_default()
            if agent is None:
                error_msg = (
                    "No agents registered. Configure at least one agent "
                    "using harness.agent() or @harness.agentloop() before handling tasks."
                )
                logger.error(
                    "Task failed: no agents registered",
                    extra={"task_id": task_id},
                )
                error_message = MessageConverter.claude_to_a2a_message(
                    role="assistant",
                    content=f"Error: {error_msg}",
                    task_id=task_id,
                    context_id=context_id,
                )
                await self.storage.update_task(
                    task_id,
                    state="failed",
                    new_messages=[error_message],  # type: ignore[arg-type]
                )
                return

            logger.debug(
                "Using agent for task",
                extra={"task_id": task_id, "agent_name": agent.config.name},
            )

            # Load existing context or create new one
            context_data = await self.storage.load_context(context_id)
            if context_data is None:
                context_data = []

            # Extract user prompt from message
            prompt = MessageConverter.extract_text_from_parts(message["parts"])

            # Build context for agent function
            message_history = [
                ContextMessage(
                    role="user" if m.get("role") == "user" else "assistant",
                    content=MessageConverter.extract_text_from_parts(m.get("parts", [])),
                )
                for m in context_data
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
            )

            # Execute agent
            if agent.func is not None:
                # Custom agent loop
                logger.debug(
                    "Executing custom agent function",
                    extra={"task_id": task_id, "agent_name": agent.config.name},
                )
                result = await agent.func(prompt, ctx, client)
            else:
                # Default behavior: simple run
                logger.debug(
                    "Executing default agent run",
                    extra={"task_id": task_id, "agent_name": agent.config.name},
                )
                result = await client.run(prompt)

            # Convert result to artifacts
            artifacts = self.build_artifacts(result)

            # Create response message
            response_message = MessageConverter.claude_to_a2a_message(
                role="assistant",
                content=result if isinstance(result, str) else str(result),
                task_id=task_id,
                context_id=context_id,
            )

            # Update context with new messages
            context_data.append(message)  # type: ignore[arg-type]
            context_data.append(response_message)
            await self.storage.update_context(context_id, context_data)

            # Update task as completed
            await self.storage.update_task(
                task_id,
                state="completed",
                new_artifacts=artifacts,
                new_messages=[response_message],  # type: ignore[arg-type]
            )

            logger.info(
                "Task completed successfully",
                extra={"task_id": task_id, "artifact_count": len(artifacts)},
            )

        except Exception as e:
            # Log full exception server-side
            logger.exception(
                "Task failed with exception",
                extra={
                    "task_id": task_id,
                    "context_id": context_id,
                    "error_type": type(e).__name__,
                },
            )

            # Return sanitized error message to client
            error_message = MessageConverter.claude_to_a2a_message(
                role="assistant",
                content=f"An error occurred: {type(e).__name__}",
                task_id=task_id,
                context_id=context_id,
            )
            await self.storage.update_task(
                task_id,
                state="failed",
                new_messages=[error_message],  # type: ignore[arg-type]
            )

        finally:
            # Remove from running tasks
            self._running_tasks.pop(task_id, None)

    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a running task."""
        task_id = params["id"]

        logger.info("Cancelling task", extra={"task_id": task_id})

        # TODO: Implement actual interruption via client.interrupt()
        # For now, just mark as canceled
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]

        await self.storage.update_task(task_id, state="canceled")

"""Client pooling for maintaining conversation history across A2A requests."""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from fastharness.logging import get_logger

logger = get_logger("client_pool")


def _hash_options(options: ClaudeAgentOptions) -> str:
    """Generate SHA256 hash of key ClaudeAgentOptions fields.

    Only includes fields that affect conversation behavior:
    - system_prompt
    - allowed_tools
    - model
    - mcp_servers
    - setting_sources
    - output_format
    """
    mcp_keys = []
    if options.mcp_servers and isinstance(options.mcp_servers, dict):
        mcp_keys = sorted(options.mcp_servers.keys())

    hash_input = (
        f"{options.system_prompt}|"
        f"{sorted(options.allowed_tools or [])}|"
        f"{options.model}|"
        f"{mcp_keys}|"
        f"{sorted(options.setting_sources or [])}|"
        f"{options.output_format}"
    )
    return hashlib.sha256(hash_input.encode()).hexdigest()


@dataclass
class ClientPoolEntry:
    """Entry in the client pool containing a ClaudeSDKClient and metadata."""

    client: ClaudeSDKClient
    context_id: str
    options_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0

    def is_stale(self, ttl_minutes: int) -> bool:
        """Check if this entry has exceeded the TTL."""
        return datetime.now(UTC) - self.last_accessed > timedelta(minutes=ttl_minutes)


class ClientPool:
    """Pool manager for long-lived ClaudeSDKClient instances keyed by context_id.

    Maintains conversation state across multiple A2A requests by reusing SDK clients.
    """

    def __init__(self, ttl_minutes: int = 15):
        """Initialize the client pool.

        Args:
            ttl_minutes: Time-to-live for idle clients (default 15 minutes)
        """
        self._pool: dict[str, ClientPoolEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl_minutes = ttl_minutes
        self._cleanup_task: asyncio.Task[None] | None = None

    async def get_or_create(
        self, context_id: str, options: ClaudeAgentOptions
    ) -> tuple[ClaudeSDKClient, bool]:
        """Get existing client or create new one for this context.

        Args:
            context_id: A2A context ID to key the client
            options: Claude agent options for this request

        Returns:
            Tuple of (client, is_new) where is_new indicates if a new client was created
        """
        async with self._lock:
            options_hash = _hash_options(options)

            # Check if we have an existing client for this context
            if context_id in self._pool:
                entry = self._pool[context_id]

                # Validate options haven't changed
                if entry.options_hash != options_hash:
                    logger.warning(
                        "Options changed mid-context, recreating client",
                        extra={"context_id": context_id},
                    )
                    await self._cleanup_entry(entry)
                else:
                    # Reuse existing client
                    entry.last_accessed = datetime.now(UTC)
                    entry.access_count += 1
                    logger.info(
                        "Reusing pooled client",
                        extra={
                            "context_id": context_id,
                            "access_count": entry.access_count,
                            "age_seconds": (datetime.now(UTC) - entry.created_at).total_seconds(),
                        },
                    )
                    return entry.client, False

            # Create new client
            logger.info("Creating new pooled client", extra={"context_id": context_id})
            client = ClaudeSDKClient(options)

            # Manually enter the client (don't use context manager)
            await client.__aenter__()

            # Store in pool
            entry = ClientPoolEntry(
                client=client,
                context_id=context_id,
                options_hash=options_hash,
            )
            self._pool[context_id] = entry

            return client, True

    async def remove(self, context_id: str) -> None:
        """Eagerly remove and cleanup a client from the pool.

        Called when a task completes or fails.
        """
        async with self._lock:
            entry = self._pool.pop(context_id, None)
            if entry:
                logger.info(
                    "Removing pooled client",
                    extra={
                        "context_id": context_id,
                        "access_count": entry.access_count,
                        "lifetime_seconds": (datetime.now(UTC) - entry.created_at).total_seconds(),
                    },
                )
                await self._cleanup_entry(entry)

    async def _cleanup_entry(self, entry: ClientPoolEntry) -> None:
        """Cleanup a single pool entry by exiting the client."""
        try:
            await entry.client.__aexit__(None, None, None)
        except Exception:
            logger.exception(
                "Error cleaning up pooled client",
                extra={"context_id": entry.context_id},
            )

    async def cleanup_stale(self) -> None:
        """Remove and cleanup entries that have exceeded the TTL."""
        async with self._lock:
            stale_contexts = [
                ctx for ctx, entry in self._pool.items() if entry.is_stale(self._ttl_minutes)
            ]

            for context_id in stale_contexts:
                entry = self._pool.pop(context_id)
                logger.info(
                    "Cleaning up stale client",
                    extra={
                        "context_id": context_id,
                        "idle_seconds": (
                            datetime.now(UTC) - entry.last_accessed
                        ).total_seconds(),
                    },
                )
                await self._cleanup_entry(entry)

    async def _cleanup_task_loop(self) -> None:
        """Background task that periodically cleans up stale entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup task")

    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_task_loop())
            logger.info("Started client pool cleanup task")

    async def shutdown(self) -> None:
        """Shutdown the pool and cleanup all clients."""
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all entries
        async with self._lock:
            logger.info("Shutting down client pool", extra={"pool_size": len(self._pool)})
            for entry in self._pool.values():
                await self._cleanup_entry(entry)
            self._pool.clear()

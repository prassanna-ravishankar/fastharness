"""Unit tests for client pooling functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from fastharness.worker.client_pool import ClientPool, _hash_options


@pytest.fixture
def base_options():
    """Create base ClaudeAgentOptions for testing."""
    return ClaudeAgentOptions(
        system_prompt="Test prompt",
        allowed_tools=["Read", "Write"],
        model="claude-sonnet-4-20250514",
        mcp_servers={"test": {"command": "test"}},
        setting_sources=["project"],
        permission_mode="bypassPermissions",
    )


@pytest.fixture
def mock_client():
    """Create mock ClaudeSDKClient."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    return client


class TestHashOptions:
    """Test options hashing for change detection."""

    def test_same_options_same_hash(self, base_options):
        """Same options produce same hash."""
        hash1 = _hash_options(base_options)
        hash2 = _hash_options(base_options)
        assert hash1 == hash2

    def test_different_system_prompt_different_hash(self, base_options):
        """Different system prompts produce different hashes."""
        hash1 = _hash_options(base_options)

        different_options = ClaudeAgentOptions(
            system_prompt="Different prompt",
            allowed_tools=base_options.allowed_tools,
            model=base_options.model,
            mcp_servers=base_options.mcp_servers,
            setting_sources=base_options.setting_sources,
        )
        hash2 = _hash_options(different_options)

        assert hash1 != hash2

    def test_different_tools_different_hash(self, base_options):
        """Different tools produce different hashes."""
        hash1 = _hash_options(base_options)

        different_options = ClaudeAgentOptions(
            system_prompt=base_options.system_prompt,
            allowed_tools=["Read", "Grep"],  # Different tools
            model=base_options.model,
            mcp_servers=base_options.mcp_servers,
            setting_sources=base_options.setting_sources,
        )
        hash2 = _hash_options(different_options)

        assert hash1 != hash2


class TestClientPool:
    """Test client pool behavior."""

    @pytest.mark.asyncio
    async def test_create_new_client(self, base_options, monkeypatch):
        """Test creating a new client in the pool."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        def mock_sdk_client(options):
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)
        client, is_new = await pool.get_or_create("ctx-1", base_options)

        assert is_new is True
        assert client == mock_client
        assert mock_client.__aenter__.called

    @pytest.mark.asyncio
    async def test_reuse_existing_client(self, base_options, monkeypatch):
        """Test reusing an existing client with same context_id."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        def mock_sdk_client(options):
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)

        # First call creates
        client1, is_new1 = await pool.get_or_create("ctx-1", base_options)
        assert is_new1 is True

        # Second call reuses
        client2, is_new2 = await pool.get_or_create("ctx-1", base_options)
        assert is_new2 is False
        assert client1 == client2

        # Should only call __aenter__ once
        assert mock_client.__aenter__.call_count == 1

    @pytest.mark.asyncio
    async def test_options_change_recreates(self, base_options, monkeypatch):
        """Test that changing options triggers client recreation."""
        call_count = 0
        clients = []

        def mock_sdk_client(options):
            nonlocal call_count
            client = MagicMock()
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock()
            clients.append(client)
            call_count += 1
            return client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)

        # First call
        client1, is_new1 = await pool.get_or_create("ctx-1", base_options)
        assert is_new1 is True

        # Change options
        different_options = ClaudeAgentOptions(
            system_prompt="Different prompt",
            allowed_tools=base_options.allowed_tools,
            model=base_options.model,
            mcp_servers=base_options.mcp_servers,
            setting_sources=base_options.setting_sources,
        )

        # Second call with different options should recreate
        client2, is_new2 = await pool.get_or_create("ctx-1", different_options)
        assert is_new2 is True
        assert client1 != client2
        assert call_count == 2

        # First client should be cleaned up
        assert clients[0].__aexit__.called

    @pytest.mark.asyncio
    async def test_remove_client(self, base_options, monkeypatch):
        """Test removing a client from the pool."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        def mock_sdk_client(options):
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)

        # Create client
        await pool.get_or_create("ctx-1", base_options)

        # Remove it
        await pool.remove("ctx-1")

        # Should call __aexit__
        assert mock_client.__aexit__.called

        # Creating again should be new
        client2, is_new = await pool.get_or_create("ctx-1", base_options)
        assert is_new is True

    @pytest.mark.asyncio
    async def test_stale_cleanup(self, base_options, monkeypatch):
        """Test that stale clients are cleaned up."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        def mock_sdk_client(options):
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        # Use very short TTL for testing
        pool = ClientPool(ttl_minutes=0)

        # Create client
        await pool.get_or_create("ctx-1", base_options)

        # Wait a bit to ensure it's stale
        await asyncio.sleep(0.1)

        # Run cleanup
        await pool.cleanup_stale()

        # Should be cleaned up
        assert mock_client.__aexit__.called
        assert "ctx-1" not in pool._pool

    @pytest.mark.asyncio
    async def test_concurrent_access(self, base_options, monkeypatch):
        """Test that concurrent access is properly serialized."""
        creation_count = 0

        def mock_sdk_client(options):
            nonlocal creation_count
            creation_count += 1
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            return mock_client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)

        # Create two concurrent requests for the same context
        results = await asyncio.gather(
            pool.get_or_create("ctx-1", base_options), pool.get_or_create("ctx-1", base_options)
        )

        # First should create, second should reuse
        assert results[0][1] is True  # is_new for first
        assert results[1][1] is False  # is_new for second
        assert creation_count == 1  # Only one client created

    @pytest.mark.asyncio
    async def test_pool_shutdown(self, base_options, monkeypatch):
        """Test that shutdown cleans up all clients."""
        clients = []

        def mock_sdk_client(options):
            client = MagicMock()
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock()
            clients.append(client)
            return client

        monkeypatch.setattr("fastharness.worker.client_pool.ClaudeSDKClient", mock_sdk_client)

        pool = ClientPool(ttl_minutes=15)

        # Create multiple clients
        await pool.get_or_create("ctx-1", base_options)
        await pool.get_or_create("ctx-2", base_options)
        await pool.get_or_create("ctx-3", base_options)

        # Shutdown
        await pool.shutdown()

        # All clients should be cleaned up
        for client in clients:
            assert client.__aexit__.called

        assert len(pool._pool) == 0

"""Tests for HarnessClient."""

import pytest

from fastharness import HarnessClient


class TestHarnessClientConfig:
    """Tests for HarnessClient configuration."""

    def test_default_values(self) -> None:
        client = HarnessClient()
        assert client.system_prompt is None
        assert client.tools == []
        assert client.model == "claude-sonnet-4-20250514"
        assert client.max_turns is None
        assert client.mcp_servers == {}
        assert client.cwd is None
        assert client.permission_mode == "bypassPermissions"

    def test_custom_values(self) -> None:
        client = HarnessClient(
            system_prompt="You are helpful",
            tools=["Read", "Write"],
            model="claude-opus-4-20250514",
            max_turns=10,
            mcp_servers={"test": {}},
            cwd="/tmp",
            permission_mode="default",
        )
        assert client.system_prompt == "You are helpful"
        assert client.tools == ["Read", "Write"]
        assert client.model == "claude-opus-4-20250514"
        assert client.max_turns == 10
        assert client.mcp_servers == {"test": {}}
        assert client.cwd == "/tmp"
        assert client.permission_mode == "default"


    def test_default_output_format(self) -> None:
        client = HarnessClient()
        assert client.output_format is None

    def test_custom_output_format(self) -> None:
        schema = {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }
        client = HarnessClient(output_format=schema)
        assert client.output_format == schema


class TestBuildOptions:
    """Tests for _build_options method."""

    def test_build_options_defaults(self) -> None:
        client = HarnessClient(
            system_prompt="Test",
            tools=["Read"],
        )
        options = client._build_options()
        assert options.system_prompt == "Test"
        assert options.allowed_tools == ["Read"]
        assert options.permission_mode == "bypassPermissions"

    def test_build_options_overrides_system_prompt(self) -> None:
        client = HarnessClient(system_prompt="Original")
        options = client._build_options(system_prompt="Override")
        assert options.system_prompt == "Override"

    def test_build_options_overrides_tools(self) -> None:
        client = HarnessClient(tools=["Read"])
        options = client._build_options(tools=["Write", "Edit"])
        assert options.allowed_tools == ["Write", "Edit"]

    def test_build_options_overrides_model(self) -> None:
        client = HarnessClient(model="claude-sonnet-4-20250514")
        options = client._build_options(model="claude-opus-4-20250514")
        assert options.model == "claude-opus-4-20250514"

    def test_build_options_overrides_max_turns(self) -> None:
        client = HarnessClient(max_turns=5)
        options = client._build_options(max_turns=10)
        assert options.max_turns == 10

    def test_build_options_overrides_permission_mode(self) -> None:
        client = HarnessClient(permission_mode="bypassPermissions")
        options = client._build_options(permission_mode="default")
        assert options.permission_mode == "default"

    def test_build_options_output_format(self) -> None:
        schema = {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"y": {"type": "string"}}},
        }
        client = HarnessClient(output_format=schema)
        options = client._build_options()
        assert options.output_format == schema

    def test_build_options_output_format_override(self) -> None:
        client = HarnessClient()
        schema = {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"z": {"type": "number"}}},
        }
        options = client._build_options(output_format=schema)
        assert options.output_format == schema

    def test_build_options_preserves_defaults(self) -> None:
        client = HarnessClient(
            system_prompt="System",
            tools=["Read"],
            max_turns=5,
        )
        # Override only one field
        options = client._build_options(tools=["Write"])
        assert options.system_prompt == "System"  # Preserved
        assert options.allowed_tools == ["Write"]  # Overridden
        assert options.max_turns == 5  # Preserved


class TestPermissionModes:
    """Tests for permission mode handling."""

    @pytest.mark.parametrize(
        "mode",
        ["default", "acceptEdits", "plan", "bypassPermissions"],
    )
    def test_valid_permission_modes(self, mode: str) -> None:
        client = HarnessClient(permission_mode=mode)  # type: ignore
        options = client._build_options()
        assert options.permission_mode == mode

"""Basic tests for FastHarness."""

import pytest
from a2a.types import TextPart

from fastharness import (
    Agent,
    AgentConfig,
    AgentContext,
    FastHarness,
    HarnessClient,
    Skill,
)
from fastharness.worker.converter import MessageConverter


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_creation(self) -> None:
        skill = Skill(id="test", name="Test", description="A test skill")
        assert skill.id == "test"
        assert skill.name == "Test"
        assert skill.description == "A test skill"
        assert skill.tags == []
        assert skill.input_modes == ["text/plain"]
        assert skill.output_modes == ["text/plain"]

    def test_skill_with_tags(self) -> None:
        skill = Skill(
            id="test",
            name="Test",
            description="A test skill",
            tags=["tag1", "tag2"],
        )
        assert skill.tags == ["tag1", "tag2"]

    def test_skill_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Skill id cannot be empty"):
            Skill(id="", name="Test", description="A test skill")

    def test_skill_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Skill name cannot be empty"):
            Skill(id="test", name="", description="A test skill")


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_agent_config_minimal(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test agent",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
        )
        assert config.name == "test"
        assert config.description == "Test agent"
        assert len(config.skills) == 1
        assert config.system_prompt is None
        assert config.tools == []
        assert config.max_turns is None

    def test_agent_config_full(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test agent",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
            system_prompt="You are helpful",
            tools=["Read", "Grep"],
            max_turns=10,
            model="claude-sonnet-4-20250514",
        )
        assert config.system_prompt == "You are helpful"
        assert config.tools == ["Read", "Grep"]
        assert config.max_turns == 10

    def test_agent_config_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentConfig name cannot be empty"):
            AgentConfig(
                name="",
                description="Test",
                skills=[Skill(id="s1", name="S1", description="Skill 1")],
            )

    def test_agent_config_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentConfig description cannot be empty"):
            AgentConfig(
                name="test",
                description="",
                skills=[Skill(id="s1", name="S1", description="Skill 1")],
            )

    def test_agent_config_empty_skills_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentConfig must have at least one skill"):
            AgentConfig(
                name="test",
                description="Test",
                skills=[],
            )

    def test_agent_config_invalid_max_turns_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentConfig max_turns must be positive"):
            AgentConfig(
                name="test",
                description="Test",
                skills=[Skill(id="s1", name="S1", description="Skill 1")],
                max_turns=0,
            )

    def test_agent_config_output_format_default(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
        )
        assert config.output_format is None

    def test_agent_config_output_format(self) -> None:
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
        }
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
            output_format=schema,
        )
        assert config.output_format == schema


class TestAgent:
    """Tests for Agent dataclass."""

    def test_agent_without_func(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
        )
        agent = Agent(config=config, func=None)
        assert agent.func is None

    def test_agent_with_func(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="Skill 1")],
        )

        async def my_func(
            prompt: str, ctx: AgentContext, client: HarnessClient
        ) -> str:
            return "result"

        agent = Agent(config=config, func=my_func)
        assert agent.func is my_func


class TestFastHarness:
    """Tests for FastHarness main class."""

    def test_harness_creation(self) -> None:
        harness = FastHarness(
            name="test-harness",
            description="Test harness",
            version="1.0.0",
        )
        assert harness.name == "test-harness"
        assert harness.description == "Test harness"
        assert harness.version == "1.0.0"

    def test_agent_registration(self) -> None:
        harness = FastHarness(name="test")
        agent = harness.agent(
            name="assistant",
            description="A test assistant",
            skills=[Skill(id="help", name="Help", description="Help users")],
            system_prompt="Be helpful",
            tools=["Read"],
        )
        assert isinstance(agent, Agent)
        assert agent.config.name == "assistant"
        assert agent.config.system_prompt == "Be helpful"
        assert agent.func is None

    def test_agent_registration_with_output_format(self) -> None:
        harness = FastHarness(name="test")
        schema = {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        }
        agent = harness.agent(
            name="structured",
            description="Structured output agent",
            skills=[Skill(id="s1", name="S1", description="S1")],
            output_format=schema,
        )
        assert agent.config.output_format == schema

    def test_agentloop_registration(self) -> None:
        harness = FastHarness(name="test")

        @harness.agentloop(
            name="custom",
            description="A custom agent",
            skills=[Skill(id="custom", name="Custom", description="Custom skill")],
        )
        async def custom_agent(
            prompt: str, ctx: AgentContext, client: HarnessClient
        ) -> str:
            return await client.run(prompt)

        assert isinstance(custom_agent, Agent)
        assert custom_agent.config.name == "custom"
        assert custom_agent.func is not None

    def test_app_creation(self) -> None:
        harness = FastHarness(name="test")
        harness.agent(
            name="assistant",
            description="Test",
            skills=[Skill(id="help", name="Help", description="Help")],
        )
        app = harness.app
        assert app is not None
        # App should be cached
        assert harness.app is app


class TestMessageConverter:
    """Tests for MessageConverter."""

    def test_extract_text_from_parts(self) -> None:
        parts = [
            {"kind": "text", "text": "Hello"},
            {"kind": "text", "text": "World"},
        ]
        result = MessageConverter.extract_text_from_parts(parts)  # type: ignore
        assert result == "Hello\nWorld"

    def test_extract_text_ignores_data_parts(self) -> None:
        parts = [
            {"kind": "text", "text": "Hello"},
            {"kind": "data", "data": {"key": "value"}},
        ]
        result = MessageConverter.extract_text_from_parts(parts)  # type: ignore
        assert result == "Hello"

    def test_text_to_artifact(self) -> None:
        artifact = MessageConverter.text_to_artifact("Result text", name="output")
        assert artifact.name == "output"
        assert len(artifact.parts) == 1
        part = artifact.parts[0].root
        assert part.kind == "text"
        assert isinstance(part, TextPart)
        assert part.text == "Result text"


class TestHarnessClient:
    """Tests for HarnessClient."""

    def test_client_creation(self) -> None:
        client = HarnessClient(
            system_prompt="Be helpful",
            tools=["Read", "Grep"],
            model="claude-sonnet-4-20250514",
        )
        assert client.system_prompt == "Be helpful"
        assert client.tools == ["Read", "Grep"]
        assert client.model == "claude-sonnet-4-20250514"

    def test_build_options(self) -> None:
        client = HarnessClient(
            system_prompt="Be helpful",
            tools=["Read"],
        )
        options = client._build_options()
        assert options.system_prompt == "Be helpful"
        assert options.allowed_tools == ["Read"]
        assert options.permission_mode == "bypassPermissions"

    def test_build_options_with_overrides(self) -> None:
        client = HarnessClient(
            system_prompt="Be helpful",
            tools=["Read"],
        )
        options = client._build_options(
            system_prompt="Override prompt",
            tools=["Write"],
        )
        assert options.system_prompt == "Override prompt"
        assert options.allowed_tools == ["Write"]


class TestAgentContext:
    """Tests for AgentContext."""

    def test_context_creation(self) -> None:
        ctx = AgentContext(
            task_id="task-123",
            context_id="ctx-456",
        )
        assert ctx.task_id == "task-123"
        assert ctx.context_id == "ctx-456"
        assert ctx.message_history == []
        assert ctx.metadata == {}
        assert ctx.deps == {}

    def test_get_last_user_message(self) -> None:
        from fastharness.core.context import Message as CtxMessage

        ctx = AgentContext(
            task_id="task-123",
            context_id="ctx-456",
            message_history=[
                CtxMessage(role="user", content="First message"),
                CtxMessage(role="assistant", content="Response"),
                CtxMessage(role="user", content="Second message"),
            ],
        )
        assert ctx.get_last_user_message() == "Second message"

    def test_get_last_user_message_empty(self) -> None:
        ctx = AgentContext(
            task_id="task-123",
            context_id="ctx-456",
        )
        assert ctx.get_last_user_message() is None

    def test_context_empty_task_id_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentContext task_id cannot be empty"):
            AgentContext(task_id="", context_id="ctx-456")

    def test_context_empty_context_id_raises(self) -> None:
        with pytest.raises(ValueError, match="AgentContext context_id cannot be empty"):
            AgentContext(task_id="task-123", context_id="")

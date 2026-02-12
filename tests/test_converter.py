"""Comprehensive tests for MessageConverter."""

from a2a.types import DataPart, Message, Part, Role, TextPart

from fastharness.worker.converter import MessageConverter


class TestClaudeToA2AParts:
    """Tests for claude_to_a2a_parts conversion."""

    def test_text_block(self) -> None:
        class MockTextBlock:
            text = "Hello world"

        parts = MessageConverter.claude_to_a2a_parts([MockTextBlock()])
        assert len(parts) == 1
        part = parts[0].root
        assert part.kind == "text"
        assert isinstance(part, TextPart)
        assert part.text == "Hello world"

    def test_tool_use_block(self) -> None:
        class MockToolUseBlock:
            id = "tool-123"
            name = "Read"
            input = {"file": "test.py"}

        parts = MessageConverter.claude_to_a2a_parts([MockToolUseBlock()])
        assert len(parts) == 1
        part = parts[0].root
        assert part.kind == "data"
        assert isinstance(part, DataPart)
        assert "tool_use" in part.data
        assert part.data["tool_use"]["name"] == "Read"
        assert part.data["tool_use"]["input"] == {"file": "test.py"}

    def test_tool_result_block(self) -> None:
        class MockToolResultBlock:
            tool_use_id = "tool-123"
            content = "File contents here"

        parts = MessageConverter.claude_to_a2a_parts([MockToolResultBlock()])
        assert len(parts) == 1
        part = parts[0].root
        assert part.kind == "data"
        assert isinstance(part, DataPart)
        assert "tool_result" in part.data
        assert part.data["tool_result"]["tool_use_id"] == "tool-123"

    def test_mixed_blocks(self) -> None:
        class MockTextBlock:
            text = "Starting analysis"

        class MockToolUseBlock:
            id = "tool-1"
            name = "Grep"
            input = {"pattern": "TODO"}

        parts = MessageConverter.claude_to_a2a_parts([MockTextBlock(), MockToolUseBlock()])
        assert len(parts) == 2
        assert parts[0].root.kind == "text"
        assert parts[1].root.kind == "data"

    def test_empty_content(self) -> None:
        parts = MessageConverter.claude_to_a2a_parts([])
        assert parts == []


class TestClaudeToA2AMessage:
    """Tests for claude_to_a2a_message conversion."""

    def test_string_content(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Hello!",
        )
        assert msg.role == Role.agent
        part = msg.parts[0].root
        assert isinstance(part, TextPart)
        assert part.text == "Hello!"

    def test_user_role(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="user",
            content="Question",
        )
        assert msg.role == Role.user

    def test_message_has_id(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Response",
        )
        assert hasattr(msg, "message_id")
        assert len(msg.message_id) > 0


class TestA2AToClaudeMessages:
    """Tests for a2a_to_claude_messages conversion."""

    def test_text_parts(self) -> None:
        history = [
            Message(
                message_id="1",
                role=Role.user,
                parts=[Part(root=TextPart(kind="text", text="Hello"))],
            )
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_agent_role_conversion(self) -> None:
        history = [
            Message(
                message_id="1",
                role=Role.agent,
                parts=[Part(root=TextPart(kind="text", text="Response"))],
            )
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)
        assert messages[0]["role"] == "assistant"

    def test_tool_use_parts(self) -> None:
        history = [
            Message(
                message_id="1",
                role=Role.agent,
                parts=[
                    Part(
                        root=DataPart(
                            kind="data",
                            data={
                                "tool_use": {
                                    "id": "tool-1",
                                    "name": "Read",
                                    "input": {"path": "/test"},
                                }
                            },
                        )
                    )
                ],
            )
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)
        assert messages[0]["content"][0]["type"] == "tool_use"
        assert messages[0]["content"][0]["name"] == "Read"

    def test_tool_result_parts(self) -> None:
        history = [
            Message(
                message_id="1",
                role=Role.user,
                parts=[
                    Part(
                        root=DataPart(
                            kind="data",
                            data={
                                "tool_result": {
                                    "tool_use_id": "tool-1",
                                    "content": "file contents",
                                }
                            },
                        )
                    )
                ],
            )
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_use_id"] == "tool-1"

    def test_empty_parts_skipped(self) -> None:
        history = [
            Message(
                message_id="1",
                role=Role.user,
                parts=[],
            )
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)
        assert len(messages) == 0


class TestTextToArtifact:
    """Tests for text_to_artifact conversion."""

    def test_basic_artifact(self) -> None:
        artifact = MessageConverter.text_to_artifact("Output text")
        assert artifact.name == "result"
        assert len(artifact.parts) == 1
        part = artifact.parts[0].root
        assert isinstance(part, TextPart)
        assert part.text == "Output text"

    def test_custom_name(self) -> None:
        artifact = MessageConverter.text_to_artifact("Data", name="output")
        assert artifact.name == "output"

    def test_artifact_has_id(self) -> None:
        artifact = MessageConverter.text_to_artifact("Test")
        assert hasattr(artifact, "artifact_id")
        assert len(artifact.artifact_id) > 0


class TestExtractTextFromParts:
    """Tests for extract_text_from_parts."""

    def test_single_text_part(self) -> None:
        parts = [Part(root=TextPart(kind="text", text="Hello"))]
        result = MessageConverter.extract_text_from_parts(parts)
        assert result == "Hello"

    def test_multiple_text_parts(self) -> None:
        parts = [
            Part(root=TextPart(kind="text", text="Line 1")),
            Part(root=TextPart(kind="text", text="Line 2")),
        ]
        result = MessageConverter.extract_text_from_parts(parts)
        assert result == "Line 1\nLine 2"

    def test_data_parts_ignored(self) -> None:
        parts = [
            Part(root=TextPart(kind="text", text="Text")),
            Part(root=DataPart(kind="data", data={"key": "value"})),
        ]
        result = MessageConverter.extract_text_from_parts(parts)
        assert result == "Text"

    def test_empty_parts(self) -> None:
        result = MessageConverter.extract_text_from_parts([])
        assert result == ""

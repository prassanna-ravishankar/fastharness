"""Comprehensive tests for MessageConverter."""

from fastharness.worker.converter import MessageConverter


class TestClaudeToA2AParts:
    """Tests for claude_to_a2a_parts conversion."""

    def test_text_block(self) -> None:
        class MockTextBlock:
            text = "Hello world"

        parts = MessageConverter.claude_to_a2a_parts([MockTextBlock()])
        assert len(parts) == 1
        assert parts[0]["kind"] == "text"
        assert parts[0]["text"] == "Hello world"

    def test_tool_use_block(self) -> None:
        class MockToolUseBlock:
            id = "tool-123"
            name = "Read"
            input = {"file": "test.py"}

        parts = MessageConverter.claude_to_a2a_parts([MockToolUseBlock()])
        assert len(parts) == 1
        assert parts[0]["kind"] == "data"
        assert "tool_use" in parts[0]["data"]
        assert parts[0]["data"]["tool_use"]["name"] == "Read"
        assert parts[0]["data"]["tool_use"]["input"] == {"file": "test.py"}

    def test_tool_result_block(self) -> None:
        class MockToolResultBlock:
            tool_use_id = "tool-123"
            content = "File contents here"

        parts = MessageConverter.claude_to_a2a_parts([MockToolResultBlock()])
        assert len(parts) == 1
        assert parts[0]["kind"] == "data"
        assert "tool_result" in parts[0]["data"]
        assert parts[0]["data"]["tool_result"]["tool_use_id"] == "tool-123"

    def test_mixed_blocks(self) -> None:
        class MockTextBlock:
            text = "Starting analysis"

        class MockToolUseBlock:
            id = "tool-1"
            name = "Grep"
            input = {"pattern": "TODO"}

        parts = MessageConverter.claude_to_a2a_parts([MockTextBlock(), MockToolUseBlock()])
        assert len(parts) == 2
        assert parts[0]["kind"] == "text"
        assert parts[1]["kind"] == "data"

    def test_empty_content(self) -> None:
        parts = MessageConverter.claude_to_a2a_parts([])
        assert parts == []


class TestClaudeToA2AMessage:
    """Tests for claude_to_a2a_message conversion."""

    def test_string_content(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Hello!",
            task_id="task-1",
            context_id="ctx-1",
        )
        assert msg["role"] == "agent"
        assert msg["parts"][0]["text"] == "Hello!"
        assert msg["task_id"] == "task-1"
        assert msg["context_id"] == "ctx-1"

    def test_user_role(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="user",
            content="Question",
        )
        assert msg["role"] == "user"

    def test_message_has_id(self) -> None:
        msg = MessageConverter.claude_to_a2a_message(
            role="assistant",
            content="Response",
        )
        assert "message_id" in msg
        assert len(msg["message_id"]) > 0


class TestA2AToClaudeMessages:
    """Tests for a2a_to_claude_messages conversion."""

    def test_text_parts(self) -> None:
        history = [
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
            }
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)  # type: ignore
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_agent_role_conversion(self) -> None:
        history = [
            {
                "role": "agent",
                "parts": [{"kind": "text", "text": "Response"}],
            }
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)  # type: ignore
        assert messages[0]["role"] == "assistant"

    def test_tool_use_parts(self) -> None:
        history = [
            {
                "role": "agent",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "tool_use": {
                                "id": "tool-1",
                                "name": "Read",
                                "input": {"path": "/test"},
                            }
                        },
                    }
                ],
            }
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)  # type: ignore
        assert messages[0]["content"][0]["type"] == "tool_use"
        assert messages[0]["content"][0]["name"] == "Read"

    def test_tool_result_parts(self) -> None:
        history = [
            {
                "role": "user",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "tool_result": {
                                "tool_use_id": "tool-1",
                                "content": "file contents",
                            }
                        },
                    }
                ],
            }
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)  # type: ignore
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_use_id"] == "tool-1"

    def test_empty_parts_skipped(self) -> None:
        history = [
            {
                "role": "user",
                "parts": [],
            }
        ]
        messages = MessageConverter.a2a_to_claude_messages(history)  # type: ignore
        assert len(messages) == 0


class TestTextToArtifact:
    """Tests for text_to_artifact conversion."""

    def test_basic_artifact(self) -> None:
        artifact = MessageConverter.text_to_artifact("Output text")
        assert artifact["name"] == "result"
        assert len(artifact["parts"]) == 1
        assert artifact["parts"][0]["text"] == "Output text"

    def test_custom_name(self) -> None:
        artifact = MessageConverter.text_to_artifact("Data", name="output")
        assert artifact["name"] == "output"

    def test_artifact_has_id(self) -> None:
        artifact = MessageConverter.text_to_artifact("Test")
        assert "artifact_id" in artifact
        assert len(artifact["artifact_id"]) > 0


class TestExtractTextFromParts:
    """Tests for extract_text_from_parts."""

    def test_single_text_part(self) -> None:
        parts = [{"kind": "text", "text": "Hello"}]
        result = MessageConverter.extract_text_from_parts(parts)  # type: ignore
        assert result == "Hello"

    def test_multiple_text_parts(self) -> None:
        parts = [
            {"kind": "text", "text": "Line 1"},
            {"kind": "text", "text": "Line 2"},
        ]
        result = MessageConverter.extract_text_from_parts(parts)  # type: ignore
        assert result == "Line 1\nLine 2"

    def test_data_parts_ignored(self) -> None:
        parts = [
            {"kind": "text", "text": "Text"},
            {"kind": "data", "data": {"key": "value"}},
        ]
        result = MessageConverter.extract_text_from_parts(parts)  # type: ignore
        assert result == "Text"

    def test_empty_parts(self) -> None:
        result = MessageConverter.extract_text_from_parts([])
        assert result == ""

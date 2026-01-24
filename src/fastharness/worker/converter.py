"""Message conversion between Claude SDK and A2A protocol."""

import uuid
from typing import Any

from fasta2a.schema import (
    Artifact,
    DataPart,
    Message,
    Part,
    TextPart,
)


class MessageConverter:
    """Convert messages between Claude SDK and A2A formats."""

    @staticmethod
    def claude_to_a2a_parts(content: list[Any]) -> list[Part]:
        """Convert Claude SDK content blocks to A2A parts."""
        parts: list[Part] = []

        for block in content:
            if hasattr(block, "text"):
                # TextBlock
                parts.append(TextPart(kind="text", text=block.text))
            elif hasattr(block, "name") and hasattr(block, "input"):
                # ToolUseBlock
                parts.append(
                    DataPart(
                        kind="data",
                        data={
                            "tool_use": {
                                "id": getattr(block, "id", ""),
                                "name": block.name,
                                "input": block.input,
                            }
                        },
                    )
                )
            elif hasattr(block, "tool_use_id") and hasattr(block, "content"):
                # ToolResultBlock
                parts.append(
                    DataPart(
                        kind="data",
                        data={
                            "tool_result": {
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                            }
                        },
                    )
                )

        return parts

    @staticmethod
    def claude_to_a2a_message(
        role: str,
        content: list[Any] | str,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> Message:
        """Convert a Claude SDK message to A2A Message."""
        if isinstance(content, str):
            parts = [TextPart(kind="text", text=content)]
        else:
            parts = MessageConverter.claude_to_a2a_parts(content)

        a2a_role: Any = "agent" if role == "assistant" else "user"
        return Message(
            role=a2a_role,
            parts=parts,
            kind="message",
            message_id=str(uuid.uuid4()),
            task_id=task_id,
            context_id=context_id,
        )

    @staticmethod
    def a2a_to_claude_messages(history: list[Message]) -> list[dict[str, Any]]:
        """Convert A2A message history to Claude SDK format."""
        messages: list[dict[str, Any]] = []

        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content_parts: list[Any] = []

            for part in msg["parts"]:
                if part["kind"] == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
                elif part["kind"] == "data" and "data" in part:
                    data = part["data"]
                    if "tool_use" in data:
                        tu = data["tool_use"]
                        content_parts.append(
                            {
                                "type": "tool_use",
                                "id": tu.get("id", ""),
                                "name": tu.get("name", ""),
                                "input": tu.get("input", {}),
                            }
                        )
                    elif "tool_result" in data:
                        tr = data["tool_result"]
                        content_parts.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.get("tool_use_id", ""),
                                "content": tr.get("content", ""),
                            }
                        )

            if content_parts:
                messages.append({"role": role, "content": content_parts})

        return messages

    @staticmethod
    def text_to_artifact(text: str, name: str = "result") -> Artifact:
        """Convert text result to an A2A Artifact."""
        return Artifact(
            artifact_id=str(uuid.uuid4()),
            name=name,
            parts=[TextPart(kind="text", text=text)],
        )

    @staticmethod
    def extract_text_from_parts(parts: list[Part]) -> str:
        """Extract plain text from A2A message parts."""
        texts = []
        for part in parts:
            if part["kind"] == "text":
                texts.append(part["text"])
        return "\n".join(texts)

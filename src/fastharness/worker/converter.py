"""Message conversion between Claude SDK and A2A protocol."""

import uuid
from typing import Any

from a2a.types import Artifact, DataPart, Message, Part, Role, TextPart


def _normalize_part(part: Any) -> dict[str, Any]:
    """Normalize a part (dict, Pydantic model, or Part wrapper) to a plain dict.

    Handles three input forms:
    - Raw dict (legacy tests): {"kind": "text", "text": "..."}
    - Part union wrapper: Part(root=TextPart(...)) -- unwrap .root first
    - Pydantic model: TextPart(kind="text", text="...") or DataPart(kind="data", data={...})
    """
    # Unwrap Part union wrapper
    actual = part.root if hasattr(part, "root") else part

    # Already a dict
    if isinstance(actual, dict):
        return actual

    # Pydantic model with .kind attribute -- convert to dict
    if hasattr(actual, "kind"):
        result: dict[str, Any] = {"kind": actual.kind}
        if actual.kind == "text" and hasattr(actual, "text"):
            result["text"] = actual.text
        elif actual.kind == "data" and hasattr(actual, "data"):
            result["data"] = actual.data
        return result

    return {}


class MessageConverter:
    """Convert messages between Claude SDK and A2A formats."""

    @staticmethod
    def claude_to_a2a_parts(content: list[Any]) -> list[Part]:
        """Convert Claude SDK content blocks to A2A parts."""
        parts: list[Part] = []

        for block in content:
            if hasattr(block, "text"):
                parts.append(Part(root=TextPart(kind="text", text=block.text)))
            elif hasattr(block, "name") and hasattr(block, "input"):
                parts.append(
                    Part(
                        root=DataPart(
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
                )
            elif hasattr(block, "tool_use_id") and hasattr(block, "content"):
                parts.append(
                    Part(
                        root=DataPart(
                            kind="data",
                            data={
                                "tool_result": {
                                    "tool_use_id": block.tool_use_id,
                                    "content": block.content,
                                }
                            },
                        )
                    )
                )

        return parts

    @staticmethod
    def claude_to_a2a_message(
        role: str,
        content: list[Any] | str,
    ) -> Message:
        """Convert a Claude SDK message to A2A Message."""
        if isinstance(content, str):
            parts: list[Part] = [Part(root=TextPart(kind="text", text=content))]
        else:
            parts = MessageConverter.claude_to_a2a_parts(content)

        a2a_role = Role.agent if role == "assistant" else Role.user
        return Message(
            role=a2a_role,
            parts=parts,
            kind="message",
            message_id=str(uuid.uuid4()),
        )

    @staticmethod
    def _convert_data_part(data: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a data part's payload to Claude SDK format.

        Returns None if the data doesn't contain a recognized tool structure.
        """
        if "tool_use" in data:
            tu = data["tool_use"]
            return {
                "type": "tool_use",
                "id": tu.get("id", ""),
                "name": tu.get("name", ""),
                "input": tu.get("input", {}),
            }
        if "tool_result" in data:
            tr = data["tool_result"]
            return {
                "type": "tool_result",
                "tool_use_id": tr.get("tool_use_id", ""),
                "content": tr.get("content", ""),
            }
        return None

    @staticmethod
    def a2a_to_claude_messages(history: list[Message]) -> list[dict[str, Any]]:
        """Convert A2A message history to Claude SDK format."""
        messages: list[dict[str, Any]] = []

        for msg in history:
            # Handle both dict (legacy test) and Pydantic model
            if isinstance(msg, dict):
                role = msg.get("role")
                parts = msg.get("parts", [])
            else:
                role = msg.role
                parts = msg.parts

            role = "user" if role == "user" else "assistant"
            content_parts: list[Any] = []

            for part in parts:
                normalized = _normalize_part(part)
                kind = normalized.get("kind")

                if kind == "text":
                    content_parts.append({"type": "text", "text": normalized.get("text", "")})
                elif kind == "data" and "data" in normalized:
                    converted = MessageConverter._convert_data_part(normalized["data"])
                    if converted is not None:
                        content_parts.append(converted)

            if content_parts:
                messages.append({"role": role, "content": content_parts})

        return messages

    @staticmethod
    def text_to_artifact(text: str, name: str = "result") -> Artifact:
        """Convert text result to an A2A Artifact."""
        return Artifact(
            artifact_id=str(uuid.uuid4()),
            name=name,
            parts=[Part(root=TextPart(kind="text", text=text))],
        )

    @staticmethod
    def extract_text_from_parts(parts: list[Part]) -> str:
        """Extract plain text from A2A message parts."""
        texts = []
        for part in parts:
            normalized = _normalize_part(part)
            if normalized.get("kind") == "text":
                texts.append(normalized.get("text", ""))
        return "\n".join(texts)

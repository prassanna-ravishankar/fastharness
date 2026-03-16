"""Message conversion between Claude SDK and A2A protocol.

A2A type construction is centralized via _make_* helpers so that
forward-incompatible changes (e.g. removal of the ``kind`` discriminator
in A2A v1.0) only need updating in one place.
"""

import uuid
from typing import Any

from a2a.types import Artifact, DataPart, Message, Part, Role, TextPart

# ---------------------------------------------------------------------------
# A2A type factories — single place to adapt when the SDK changes
# ---------------------------------------------------------------------------

_TEXT_PART_NEEDS_KIND = "kind" in TextPart.model_fields


def _text_part(text: str) -> Part:
    kwargs: dict[str, Any] = {"text": text}
    if _TEXT_PART_NEEDS_KIND:
        kwargs["kind"] = "text"
    return Part(root=TextPart(**kwargs))


def _data_part(data: dict[str, Any]) -> Part:
    kwargs: dict[str, Any] = {"data": data}
    if _TEXT_PART_NEEDS_KIND:  # DataPart follows same pattern
        kwargs["kind"] = "data"
    return Part(root=DataPart(**kwargs))


_MESSAGE_NEEDS_KIND = "kind" in Message.model_fields


def _message(role: Role, parts: list[Part], message_id: str) -> Message:
    kwargs: dict[str, Any] = {"role": role, "parts": parts, "message_id": message_id}
    if _MESSAGE_NEEDS_KIND:
        kwargs["kind"] = "message"
    return Message(**kwargs)


# ---------------------------------------------------------------------------
# Part normalization (reading parts from any source)
# ---------------------------------------------------------------------------


def _normalize_part(part: Any) -> dict[str, Any]:
    """Normalize a part (dict, Pydantic model, or Part wrapper) to a plain dict.

    Handles three input forms:
    - Raw dict (legacy tests): {"kind": "text", "text": "..."}
    - Part union wrapper: Part(root=TextPart(...)) -- unwrap .root first
    - Pydantic model: TextPart(text="...") or DataPart(data={...})

    For v1.0 compatibility, detects the part type by checking for
    characteristic attributes rather than relying on ``kind``.
    """
    actual = part.root if hasattr(part, "root") else part

    if isinstance(actual, dict):
        # Ensure a synthetic kind for v1.0 dicts that lack it
        if "kind" not in actual:
            if "text" in actual:
                actual = {**actual, "kind": "text"}
            elif "data" in actual:
                actual = {**actual, "kind": "data"}
        return actual

    # Pydantic model — detect type by attribute, not kind
    if hasattr(actual, "text") and isinstance(getattr(actual, "text", None), str):
        return {"kind": "text", "text": actual.text}
    if hasattr(actual, "data"):
        return {"kind": "data", "data": actual.data}

    # Fall back to kind if present (0.3.x compat)
    kind = getattr(actual, "kind", None)
    if kind == "text" and hasattr(actual, "text"):
        return {"kind": "text", "text": actual.text}
    if kind == "data" and hasattr(actual, "data"):
        return {"kind": "data", "data": actual.data}

    return {}


class MessageConverter:
    """Convert messages between Claude SDK and A2A formats."""

    @staticmethod
    def claude_to_a2a_parts(content: list[Any]) -> list[Part]:
        """Convert Claude SDK content blocks to A2A parts."""
        parts: list[Part] = []

        for block in content:
            if hasattr(block, "text"):
                parts.append(_text_part(block.text))
            elif hasattr(block, "name") and hasattr(block, "input"):
                parts.append(
                    _data_part(
                        {
                            "tool_use": {
                                "id": getattr(block, "id", ""),
                                "name": block.name,
                                "input": block.input,
                            }
                        }
                    )
                )
            elif hasattr(block, "tool_use_id") and hasattr(block, "content"):
                parts.append(
                    _data_part(
                        {
                            "tool_result": {
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                            }
                        }
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
            parts: list[Part] = [_text_part(content)]
        else:
            parts = MessageConverter.claude_to_a2a_parts(content)

        a2a_role = Role.agent if role == "assistant" else Role.user
        return _message(a2a_role, parts, str(uuid.uuid4()))

    @staticmethod
    def _convert_data_part(data: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a data part's payload to Claude SDK format."""
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
            parts=[_text_part(text)],
        )

    @staticmethod
    def extract_text_from_parts(parts: list[Part] | None) -> str:
        """Extract plain text from A2A message parts."""
        if not parts:
            return ""
        texts = []
        for part in parts:
            normalized = _normalize_part(part)
            if normalized.get("kind") == "text":
                texts.append(normalized.get("text", ""))
        return "\n".join(texts)

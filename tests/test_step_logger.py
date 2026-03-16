"""Tests for step logging."""

import logging

import pytest

from fastharness.step_logger import ConsoleStepLogger, StepEvent, StepLogger


class TestStepEvent:
    def test_creation(self) -> None:
        event = StepEvent(step_type="tool_call", turn_number=0, data={"name": "Read"})
        assert event.step_type == "tool_call"
        assert event.turn_number == 0
        assert event.data == {"name": "Read"}

    def test_all_step_types(self) -> None:
        for st in ("tool_call", "assistant_message", "turn_complete"):
            event = StepEvent(step_type=st, turn_number=1, data={})
            assert event.step_type == st


class TestConsoleStepLogger:
    @pytest.mark.asyncio
    async def test_log_tool_call(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleStepLogger()
        event = StepEvent(
            step_type="tool_call",
            turn_number=0,
            data={"name": "Read", "id": "call_123", "input": {}},
        )
        with caplog.at_level(logging.INFO):
            await logger.log_step(event)
        assert "Tool call" in caplog.text
        assert "Read" in caplog.text

    @pytest.mark.asyncio
    async def test_log_assistant_message(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleStepLogger()
        event = StepEvent(
            step_type="assistant_message",
            turn_number=1,
            data={"text": "Hello world"},
        )
        with caplog.at_level(logging.INFO):
            await logger.log_step(event)
        assert "Assistant message" in caplog.text
        assert "Hello world" in caplog.text

    @pytest.mark.asyncio
    async def test_log_assistant_message_truncates(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleStepLogger()
        long_text = "x" * 200
        event = StepEvent(
            step_type="assistant_message",
            turn_number=0,
            data={"text": long_text},
        )
        with caplog.at_level(logging.INFO):
            await logger.log_step(event)
        # Text preview should be truncated to 100 chars
        assert len(long_text[:100]) == 100

    @pytest.mark.asyncio
    async def test_log_turn_complete(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleStepLogger()
        event = StepEvent(
            step_type="turn_complete",
            turn_number=2,
            data={"cost_usd": 0.01, "usage": {"input_tokens": 100}},
        )
        with caplog.at_level(logging.INFO):
            await logger.log_step(event)
        assert "Turn complete" in caplog.text

    @pytest.mark.asyncio
    async def test_log_unknown_step_type(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = ConsoleStepLogger()
        event = StepEvent(
            step_type="unknown_type",  # type: ignore[arg-type]
            turn_number=0,
            data={},
        )
        with caplog.at_level(logging.WARNING):
            await logger.log_step(event)
        assert "Unknown step type" in caplog.text

    def test_fmt_filters_none(self) -> None:
        result = ConsoleStepLogger._fmt({"a": 1, "b": None, "c": "x"})
        assert "a=1" in result
        assert "c=x" in result
        assert "b" not in result

    def test_fmt_empty(self) -> None:
        result = ConsoleStepLogger._fmt({})
        assert result == ""

    def test_implements_protocol(self) -> None:
        assert isinstance(ConsoleStepLogger(), StepLogger)

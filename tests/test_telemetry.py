"""Tests for telemetry and cost tracking."""

import pytest

from fastharness.telemetry import CostTracker, ExecutionMetrics, TelemetryCallback


def _make_metrics(**overrides) -> ExecutionMetrics:
    defaults = dict(
        task_id="task-1",
        session_id="sess-1",
        total_cost_usd=0.01,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        duration_ms=500,
        duration_api_ms=400,
        num_turns=1,
        status="success",
    )
    defaults.update(overrides)
    return ExecutionMetrics(**defaults)


class TestExecutionMetrics:
    def test_creation(self) -> None:
        m = _make_metrics()
        assert m.task_id == "task-1"
        assert m.status == "success"
        assert m.timestamp is not None

    def test_error_status(self) -> None:
        m = _make_metrics(status="error")
        assert m.status == "error"

    def test_none_cost(self) -> None:
        m = _make_metrics(total_cost_usd=None)
        assert m.total_cost_usd is None


class TestCostTracker:
    @pytest.mark.asyncio
    async def test_tracks_cost(self) -> None:
        tracker = CostTracker()
        await tracker.on_complete(_make_metrics(total_cost_usd=0.05))
        assert tracker.total_cost_usd == pytest.approx(0.05)
        assert len(tracker.executions) == 1

    @pytest.mark.asyncio
    async def test_accumulates_cost(self) -> None:
        tracker = CostTracker()
        await tracker.on_complete(_make_metrics(total_cost_usd=0.03))
        await tracker.on_complete(_make_metrics(total_cost_usd=0.07))
        assert tracker.total_cost_usd == pytest.approx(0.10)
        assert len(tracker.executions) == 2

    @pytest.mark.asyncio
    async def test_none_cost_ignored(self) -> None:
        tracker = CostTracker()
        await tracker.on_complete(_make_metrics(total_cost_usd=None))
        assert tracker.total_cost_usd == 0.0
        assert len(tracker.executions) == 1

    @pytest.mark.asyncio
    async def test_warn_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = CostTracker(warn_threshold_usd=0.05)
        await tracker.on_complete(_make_metrics(total_cost_usd=0.06))
        assert "warning" in caplog.text.lower() or tracker.total_cost_usd > 0.05

    @pytest.mark.asyncio
    async def test_error_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = CostTracker(error_threshold_usd=0.10)
        await tracker.on_complete(_make_metrics(total_cost_usd=0.11))
        assert tracker.total_cost_usd > 0.10

    @pytest.mark.asyncio
    async def test_below_threshold_no_warning(self) -> None:
        tracker = CostTracker(warn_threshold_usd=1.0)
        await tracker.on_complete(_make_metrics(total_cost_usd=0.01))
        # Should not raise or log warning
        assert tracker.total_cost_usd == pytest.approx(0.01)

    def test_implements_protocol(self) -> None:
        assert isinstance(CostTracker(), TelemetryCallback)

    @pytest.mark.asyncio
    async def test_custom_thresholds(self) -> None:
        tracker = CostTracker(warn_threshold_usd=5.0, error_threshold_usd=50.0)
        assert tracker.warn_threshold_usd == 5.0
        assert tracker.error_threshold_usd == 50.0

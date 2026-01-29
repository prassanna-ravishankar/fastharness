"""Cost tracking and telemetry for agent execution."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from fastharness.logging import get_logger

logger = get_logger("telemetry")


@dataclass
class ExecutionMetrics:
    """Metrics captured from ResultMessage after execution."""

    task_id: str
    session_id: str
    total_cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    status: Literal["success", "error"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class TelemetryCallback(Protocol):
    """Protocol for telemetry callbacks."""

    async def on_complete(self, metrics: ExecutionMetrics) -> None:
        """Called when execution completes with metrics."""
        ...


class CostTracker(TelemetryCallback):
    """Built-in cost tracker with threshold warnings."""

    def __init__(
        self,
        warn_threshold_usd: float = 1.0,
        error_threshold_usd: float = 10.0,
    ):
        self.warn_threshold_usd = warn_threshold_usd
        self.error_threshold_usd = error_threshold_usd
        self.total_cost_usd = 0.0
        self.executions: list[ExecutionMetrics] = []

    async def on_complete(self, metrics: ExecutionMetrics) -> None:
        """Track cost and emit warnings if thresholds exceeded."""
        self.executions.append(metrics)

        if metrics.total_cost_usd is not None:
            self.total_cost_usd += metrics.total_cost_usd

            if self.total_cost_usd > self.error_threshold_usd:
                logger.error(
                    f"Cost threshold exceeded: ${self.total_cost_usd:.4f}",
                    extra={"threshold": self.error_threshold_usd},
                )
            elif self.total_cost_usd > self.warn_threshold_usd:
                logger.warning(
                    f"Cost threshold warning: ${self.total_cost_usd:.4f}",
                    extra={"threshold": self.warn_threshold_usd},
                )

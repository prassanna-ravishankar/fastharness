"""OpenTelemetry GenAI tracing for FastHarness."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from fastharness.logging import get_logger

logger = get_logger("tracing")

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


PROVIDER_NAME = "anthropic"
TRACER_NAME = "fastharness"


def _get_tracer() -> Any | None:
    if not _HAS_OTEL:
        return None
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def agent_span(agent_name: str, model: str):
    """invoke_agent span wrapping full task execution."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": PROVIDER_NAME,
            "gen_ai.agent.name": agent_name,
            "gen_ai.request.model": model,
        },
        end_on_exit=False,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__qualname__)
            raise
        finally:
            span.end()


@contextmanager
def chat_span(model: str):
    """chat span wrapping a single HarnessClient.run()/stream() call."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": PROVIDER_NAME,
            "gen_ai.request.model": model,
        },
        end_on_exit=False,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__qualname__)
            raise
        finally:
            span.end()


@contextmanager
def tool_span(tool_name: str, tool_call_id: str = ""):
    """execute_tool span for each tool invocation."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        f"execute_tool {tool_name}",
        kind=SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": tool_call_id,
        },
    ) as span:
        yield span


def set_span_attributes(span: Any, **attrs: Any) -> None:
    """Set attributes on span, skipping None values."""
    if span is None:
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def record_usage(span: Any, usage: dict[str, Any] | None) -> None:
    """Set token usage attributes on a span from ResultMessage usage."""
    if span is None or usage is None:
        return
    set_span_attributes(
        span,
        **{
            "gen_ai.usage.input_tokens": usage.get("input_tokens"),
            "gen_ai.usage.output_tokens": usage.get("output_tokens"),
        },
    )

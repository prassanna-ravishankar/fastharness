"""Tests for OpenTelemetry tracing integration."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from fastharness import AgentConfig, FastHarness, Skill
from fastharness.tracing import (
    _HAS_OTEL,
    agent_span,
    chat_span,
    record_usage,
    set_span_attributes,
    tool_span,
)


class _InMemoryExporter(SpanExporter):
    """Collects spans in a list for test assertions."""

    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()


@pytest.fixture(autouse=True)
def otel_spans():
    """Set up in-memory OTel exporter and return exporter for assertions.

    Uses a single TracerProvider for all tests in a session to avoid the
    "Overriding of current TracerProvider is not allowed" warning. Each test
    gets a fresh exporter via a new processor.
    """
    exporter = _InMemoryExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    # Force override the global provider for tests
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()


class TestOtelAvailable:
    def test_has_otel(self) -> None:
        assert _HAS_OTEL is True


class TestAgentSpan:
    def test_agent_span_creates_span(self, otel_spans: _InMemoryExporter) -> None:
        with agent_span("my-agent", "claude-sonnet-4-20250514") as span:
            assert span is not None

        assert len(otel_spans.spans) == 1
        s = otel_spans.spans[0]
        assert s.name == "invoke_agent my-agent"
        assert s.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert s.attributes["gen_ai.provider.name"] == "anthropic"
        assert s.attributes["gen_ai.agent.name"] == "my-agent"
        assert s.attributes["gen_ai.request.model"] == "claude-sonnet-4-20250514"

    def test_agent_span_error_sets_status(self, otel_spans: _InMemoryExporter) -> None:
        with pytest.raises(ValueError, match="test error"):
            with agent_span("fail-agent", "model"):
                raise ValueError("test error")

        assert len(otel_spans.spans) == 1
        s = otel_spans.spans[0]
        assert s.status.status_code == trace.StatusCode.ERROR
        assert s.attributes["error.type"] == "ValueError"


class TestChatSpan:
    def test_chat_span_creates_span(self, otel_spans: _InMemoryExporter) -> None:
        with chat_span("claude-sonnet-4-20250514") as span:
            assert span is not None

        assert len(otel_spans.spans) == 1
        s = otel_spans.spans[0]
        assert s.name == "chat claude-sonnet-4-20250514"
        assert s.attributes["gen_ai.operation.name"] == "chat"
        assert s.attributes["gen_ai.provider.name"] == "anthropic"
        assert s.attributes["gen_ai.request.model"] == "claude-sonnet-4-20250514"

    def test_chat_span_error(self, otel_spans: _InMemoryExporter) -> None:
        with pytest.raises(RuntimeError):
            with chat_span("model"):
                raise RuntimeError("boom")

        assert len(otel_spans.spans) == 1
        assert otel_spans.spans[0].attributes["error.type"] == "RuntimeError"


class TestToolSpan:
    def test_tool_span_creates_span(self, otel_spans: _InMemoryExporter) -> None:
        with tool_span("Read", "toolu_abc") as span:
            assert span is not None

        assert len(otel_spans.spans) == 1
        s = otel_spans.spans[0]
        assert s.name == "execute_tool Read"
        assert s.attributes["gen_ai.operation.name"] == "execute_tool"
        assert s.attributes["gen_ai.tool.name"] == "Read"
        assert s.attributes["gen_ai.tool.call.id"] == "toolu_abc"


class TestSpanHierarchy:
    def test_nested_spans(self, otel_spans: _InMemoryExporter) -> None:
        with agent_span("test-agent", "model"):
            with chat_span("model"):
                with tool_span("Read", "t1"):
                    pass
                with tool_span("Grep", "t2"):
                    pass

        assert len(otel_spans.spans) == 4

        span_by_name = {s.name: s for s in otel_spans.spans}
        agent_s = span_by_name["invoke_agent test-agent"]
        chat_s = span_by_name["chat model"]
        read_s = span_by_name["execute_tool Read"]
        grep_s = span_by_name["execute_tool Grep"]

        # chat is child of agent
        assert chat_s.parent.span_id == agent_s.context.span_id
        # tools are children of chat
        assert read_s.parent.span_id == chat_s.context.span_id
        assert grep_s.parent.span_id == chat_s.context.span_id


class TestHelpers:
    def test_set_span_attributes_none_span(self) -> None:
        set_span_attributes(None, key="value")

    def test_set_span_attributes_skips_none(self, otel_spans: _InMemoryExporter) -> None:
        with agent_span("test", "model") as span:
            set_span_attributes(span, present="yes", missing=None)

        assert otel_spans.spans[0].attributes.get("present") == "yes"
        assert "missing" not in otel_spans.spans[0].attributes

    def test_record_usage(self, otel_spans: _InMemoryExporter) -> None:
        with chat_span("model") as span:
            record_usage(span, {"input_tokens": 100, "output_tokens": 50})

        assert otel_spans.spans[0].attributes["gen_ai.usage.input_tokens"] == 100
        assert otel_spans.spans[0].attributes["gen_ai.usage.output_tokens"] == 50

    def test_record_usage_none_span(self) -> None:
        record_usage(None, {"input_tokens": 100})

    def test_record_usage_none_usage(self, otel_spans: _InMemoryExporter) -> None:
        with chat_span("model") as span:
            record_usage(span, None)

        assert "gen_ai.usage.input_tokens" not in otel_spans.spans[0].attributes


class TestTracingFlag:
    def test_agent_config_tracing_default(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
        )
        assert config.tracing is False

    def test_agent_config_tracing_true(self) -> None:
        config = AgentConfig(
            name="test",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
            tracing=True,
        )
        assert config.tracing is True

    def test_harness_tracing_inheritance(self) -> None:
        harness = FastHarness(name="test", tracing=True)
        agent = harness.agent(
            name="inherited",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
        )
        assert agent.config.tracing is True

    def test_harness_tracing_override(self) -> None:
        harness = FastHarness(name="test", tracing=True)
        agent = harness.agent(
            name="overridden",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
            tracing=False,
        )
        assert agent.config.tracing is False

    def test_harness_default_no_tracing(self) -> None:
        harness = FastHarness(name="test")
        agent = harness.agent(
            name="default",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
        )
        assert agent.config.tracing is False

    def test_agentloop_tracing_inheritance(self) -> None:
        harness = FastHarness(name="test", tracing=True)

        @harness.agentloop(
            name="loop",
            description="Test",
            skills=[Skill(id="s1", name="S1", description="S1")],
        )
        async def loop_agent(prompt, ctx, client):
            return ""

        assert loop_agent.config.tracing is True

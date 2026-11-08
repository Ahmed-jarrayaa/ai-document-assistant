import functools
import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "ai-document-assistant")
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

# The tracer is what code uses to create spans. Before setup_telemetry() runs,
# it does nothing (no-op), so importing modules never breaks anything.
tracer = trace.get_tracer(SERVICE_NAME)

_initialized = False
def tracing_enabled() -> bool:
    """Read at call time, so tests (or you) can switch tracing off via the environment."""
    return os.getenv("TRACING_ENABLED", "true").lower() == "true"


def setup_telemetry() -> None:
    """Configure OpenTelemetry once per process (call from an entry point)."""
    global _initialized
    if _initialized or not tracing_enabled():
        return

    # Resource = identity of this service (shown in Jaeger's "Service" dropdown)
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))

    # Exporter = how spans leave the app. BatchSpanProcessor sends them in the
    # background in batches, so tracing does not slow down requests.
    exporter = OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # OPTIONAL: automatic SQL instrumentation (needs opentelemetry-instrumentation-psycopg)
    if os.getenv("OTEL_PSYCOPG_AUTO", "false").lower() == "true":
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument()

    _initialized = True


def flush_telemetry() -> None:
    """Send pending spans now. Short scripts must call this before exiting."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def traced(span_name: str):
    """Decorator: run the function inside a span.

    Exceptions are recorded on the span automatically and the span is marked
    ERROR, then the exception continues to propagate.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
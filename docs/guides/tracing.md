# Trace with nodeartifact

[nodeartifact](https://nodestep-ai.github.io/nodestep/nodeartifact/) records nodestep runs as OpenTelemetry spans and shows them in a local web UI.

## When you need it

Use it to see each run, node, model call and tool call as a trace.

## Steps

1. Add nodeartifact from GitHub, with the `server` extra for the viewer: `uv add "nodeartifact[server] @ git+https://github.com/nodestep-ai/nodeartifact"`. It is a separate package and not on PyPI yet; nodestep itself has no OpenTelemetry dependency.
2. Start the server with `uv run nodeartifact serve`. It listens on `127.0.0.1:4318`, the standard OTLP/HTTP port, and the viewer is at `http://127.0.0.1:4318/`.
3. `nodeartifact.configure(endpoint)` returns an OpenTelemetry `TracerProvider` that sends spans to the server. It does not set the global provider, so pass it to `instrument`.
4. Wrap the graph with `nodeartifact.instrument(graph, tracer_provider=...)`, run the copy it returns, then call `provider.shutdown()` to send the last spans before the script exits.

## Complete example

<!-- not-run -->

```python
import asyncio

import nodeartifact
from nodestep import ScriptedChat, build_react_agent, run_agent

agent = build_react_agent(ScriptedChat(["Order A-1001 has shipped."]), tools=[])

provider = nodeartifact.configure("http://127.0.0.1:4318", service_name="orders")
traced = nodeartifact.instrument(agent, tracer_provider=provider)
print(asyncio.run(run_agent(traced, "Where is order A-1001?", thread_id="order-1")))
provider.shutdown()
```

It prints `Order A-1001 has shipped.`, and the run appears in the viewer as one trace.

## In tests

Tests need no server. Use a `TracerProvider` with OpenTelemetry's `InMemorySpanExporter`, run with `ScriptedChat`, and check the finished spans:

<!-- not-run -->

```python
import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import nodeartifact
from nodestep import ScriptedChat, build_react_agent, run_agent

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

agent = build_react_agent(ScriptedChat(["Hello."]), tools=[])
traced = nodeartifact.instrument(agent, tracer_provider=provider)
asyncio.run(run_agent(traced, "Hi", thread_id="test-1"))

print([span.name for span in exporter.get_finished_spans()])
```

```text
['chat scripted', 'nodestep.node think', 'nodestep.graph agent']
```

## Good to know

- `instrument` returns a copy of the graph with a `TracingMiddleware` after its own middleware; the original graph is not changed. `TracingMiddleware` turns the graph, node, model and tool [middleware](../concepts/middleware.md) hooks into spans. Each run is one trace, with a span for the run, each node task, each model call and each tool call.
- Child graphs, sub-agents and background agents are traced when you instrument their graphs too. The [nodeartifact docs](https://nodestep-ai.github.io/nodestep/nodeartifact/) list the spans and their attributes, the web UI and the server options.

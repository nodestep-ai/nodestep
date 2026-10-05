# nodestep

nodestep is a Python library for LLM agents and workflows built as graphs: nodes are plain functions, async or sync, that read a typed state and return an update. You declare how they connect, and nodestep runs them step by step, can store every step and can pause for a person.

!!! warning "In development"
    Alpha (0.1.0a1). Anything may change between releases without a deprecation period, so pin a tag or a commit, as the [Install](install.md#pin-a-version) page shows. nodestep is not on PyPI yet.

## Install

```bash
uv add "nodestep @ git+https://github.com/nodestep-ai/nodestep"
```

[Install](install.md) covers the extras and how to pin a version.

## A first graph

```python
from nodestep import END, START, BaseState, Graph, node


class Greeting(BaseState):
    name: str = ""
    text: str = ""


@node
def greet(state: Greeting) -> dict:
    return {"text": f"Hello, {state.name}!"}


graph = Graph(Greeting).flow(START >> greet, greet >> END)

print(graph.invoke({"name": "Ada"}).state.text)
```

```text
Hello, Ada!
```

The state holds the data, nodes do the work, and the flow says what runs next. The [Quickstart](quickstart.md) builds a larger graph step by step and needs no API key.

## Core concepts

- [Graphs and flow](concepts/graphs.md): how nodes connect, how a run is routed, and how to draw the graph.
- [State](concepts/state.md): the typed data a graph works on, and the reducers that merge updates into it.
- [Nodes and context](concepts/nodes.md): what a node receives and what it can return.
- [Running](concepts/execution.md): `invoke()` and `stream()`, supersteps, limits and timeouts.

## Capabilities

- [Streaming](concepts/streaming.md): per-task updates, the full state, custom events, model tokens and debug events while the graph runs.
- [Persistence and time travel](concepts/persistence.md): threads, history and forks, in memory or on disk.
- [Interrupts](concepts/interrupts.md): a node pauses for a person and continues with the answer.
- [Agents and tools](concepts/agents.md): chat models through the `Chat` protocol, built-in chat integrations, `ScriptedChat` for tests, tools from typed functions and a ready-made agent.
- [Middleware](concepts/middleware.md): hooks around graphs, nodes, tools and model calls, and built-in middleware for tool approvals, tool-call limits, summarization, to-do lists, skills and memory.
- [Sub-agents](concepts/sub-agents.md): other graphs that a node runs, inside its step or in the background.
- [Workspace and memory](concepts/workspace.md): files for tools, on disk or in memory, and long-term memory for a model.

## How-to

- [Branch on the state](guides/branching.md), [Fan out with Send](guides/fan-out.md) and [Pause for approval](guides/approval.md).
- [Write a chat integration](guides/chat-models.md) and [Test with ScriptedChat](guides/testing.md).
- [Trace with nodeartifact](guides/tracing.md): runs recorded as OpenTelemetry spans and shown in a local web UI.
- [Use the sandbox](guides/sandbox.md), in the `sandbox` extra: a local web page that runs one graph, shows each step and answers interrupts.
- [Run the examples](examples.md).

## License

MIT. The source is on [GitHub](https://github.com/nodestep-ai/nodestep).

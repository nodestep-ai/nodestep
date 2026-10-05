# Streaming

`stream()` and `astream()` yield events while the graph runs, so you can show progress, intermediate state and model tokens as they arrive.

```python
from nodestep import END, START, BaseState, Graph, NodeContext, node


class Upload(BaseState):
    files: list[str] = []
    done: int = 0


@node
def process(state: Upload, ctx: NodeContext) -> dict:
    for index, name in enumerate(state.files, start=1):
        ctx.emit({"file": name, "progress": f"{index}/{len(state.files)}"})
    return {"done": len(state.files)}


graph = Graph(Upload).flow(START >> process, process >> END)

for event in graph.stream(
    {"files": ["a.csv", "b.csv"]}, stream_mode=["custom", "values"]
):
    print(event.mode, event.data if event.mode != "final" else event.data.state)
```

```text
custom {'file': 'a.csv', 'progress': '1/2'}
custom {'file': 'b.csv', 'progress': '2/2'}
values {'files': ['a.csv', 'b.csv'], 'done': 2}
final {'files': ['a.csv', 'b.csv'], 'done': 2}
```

The node sends two `"custom"` events while it works. After the superstep comes a `"values"` event with the state, and the stream ends with `"final"`.

## Pick what to stream

`stream_mode` picks the events you get. Pass one mode name, or a list of them. Each event is a `StreamEvent` with a `mode` and its `data`.

| Mode | When | `data` |
|---|---|---|
| `"updates"` | After each superstep, one event per task | The task's update dict, or `None` |
| `"values"` | After each superstep | The merged state as a dict |
| `"custom"` | When a node calls `ctx.emit(data)` | The emitted value |
| `"tokens"` | While a model node streams | A `ChatStreamChunk` |
| `"debug"` | Around every task and merge | A dict with a `"type"`: `"node_input"`, `"node_output"`, `"routes"` or `"state_merge"` |

In async code, use `astream()` with `async for`. It takes the same arguments as `stream()`.

## The last event

Every stream ends with one terminal event. It is `"final"` when the run completed and `"interrupt"` when it paused. You get it whatever modes you pick, so `stream_mode=[]` yields only that event.

```python
from nodestep import END, START, BaseState, Graph, InMemoryStateStore, interrupt, node


class Refund(BaseState):
    amount: float = 0.0
    approved: bool = False


@node
def review(state: Refund) -> dict:
    answer = interrupt(f"Approve a refund of {state.amount}?", id="approve")
    return {"approved": answer == "yes"}


graph = Graph(Refund, state_store=InMemoryStateStore()).flow(
    START >> review,
    review >> END,
)

for event in graph.stream({"amount": 80}, thread_id="refund-1", stream_mode=[]):
    print(event.mode, list(event.data.interrupts))
```

```text
interrupt ['review:approve']
```

A `"final"` event carries `FinalEventData` with `state` and `thread_id`. An `"interrupt"` event carries `InterruptEventData` with `state`, `thread_id` and `interrupts`; see [Interrupts](interrupts.md).

## Event fields

Besides `mode` and `data`, every event carries `node`, `step`, `task_id`, `run_id` and `checkpoint_id`.

```python
from nodestep import END, START, BaseState, Graph, node


class Ticket(BaseState):
    text: str = ""
    words: int = 0
    summary: str = ""


@node
def count_words(state: Ticket) -> dict:
    return {"words": len(state.text.split())}


@node
def summarize(state: Ticket) -> dict:
    return {"summary": f"{state.words} words"}


graph = Graph(Ticket).flow(
    START >> count_words,
    count_words >> summarize,
    summarize >> END,
)

for event in graph.stream({"text": "Checkout is down"}, stream_mode="updates"):
    print(event.mode, event.step, event.node, event.task_id)
```

```text
updates 0 count_words count_words
updates 1 summarize summarize
final 1 None None
```

- `step` is the superstep, counted from 0.
- `node` and `task_id` name the task the event belongs to. Events about a whole superstep, and the terminal events, have neither.
- `checkpoint_id` points at the stored history, so you can load the state the event saw; see [Persistence and time travel](persistence.md).

## Custom events

A node sends its own events with `ctx.emit(data)`, as the opening example does for progress. The value can be anything, and it reaches consumers that ask for `"custom"`.

## Model tokens

A model node streams the model's answer when you build it with `stream=True`. Its chunks reach consumers that ask for `"tokens"`.

```python
import asyncio

from nodestep import ScriptedChat, build_react_agent
from nodestep.chat import HumanMessage

agent = build_react_agent(
    ScriptedChat(["Your order has shipped."]), tools=[], stream=True
)


async def main() -> None:
    async for event in agent.astream(
        {"messages": [HumanMessage(content="Where is my order?")]},
        stream_mode="tokens",
    ):
        if event.mode == "tokens" and event.data.content_delta:
            print(repr(event.data.content_delta))


asyncio.run(main())
```

```text
'Your'
' order'
' has'
' shipped.'
```

## Closing a stream early

Leaving the loop early closes the stream and stops the run. With a state store, you can continue the run later; see [Continue an unfinished superstep](persistence.md#continue-an-unfinished-superstep).

While your loop body handles an event, the graph waits for it.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| `stream_mode` | It is required and has no default. Pass a mode name or a list of mode names |
| Terminal event | Every stream ends with exactly one: `"final"` when the run completed, `"interrupt"` when it paused |
| `stream_mode=[]` | Yields only the terminal event |
| Order of `"updates"` events | One event per task of the superstep, in scheduling order |
| `"updates"` data | The task's update dict, or `None` when the task returned no update |
| `"values"` data | The merged state as a dict, after each superstep |
| `"debug"` data | A dict whose `"type"` is `"node_input"`, `"node_output"`, `"routes"` or `"state_merge"` |
| `node` and `task_id` | `None` for `"values"` events, the `"state_merge"` debug event and the terminal events |
| `step` | The superstep of the run, counted from 0 |
| `checkpoint_id` | The history event to load the event's state at: `superstep_completed` for `"updates"` and `"values"`, the run's last event for the terminal events, `None` without a state store |
| `ctx.emit(data)` | Sends only `"custom"` events, with the emitted value as `data` |
| State in events | Events carry copies of the state: changing them does not change the run |
| Model tokens | A model node streams only when built with `stream=True` (`model_node(stream=True)`), whatever the caller's stream mode. Its `ChatStreamChunk`s go to consumers that ask for `"tokens"` |
| Leaving the loop early | Closes the stream and stops the run. With a state store, the running superstep stays unfinished, and `invoke(None, thread_id=...)` continues it |
| Time in the loop body | With `stream()`, the graph waits while your loop body handles an event. Time on `"custom"` or `"tokens"` events, which arrive during a superstep, counts toward the graph and node timeouts. Time on `"updates"` and `"values"` events, between supersteps, does not |

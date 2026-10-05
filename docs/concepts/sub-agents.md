# Sub-agents

A sub-agent is another graph that your graph runs on behalf of one of its nodes.

```python
import asyncio

from pydantic import Field

from nodestep import END, START, AgentTask, BaseState, Graph, NodeContext, node


class Research(BaseState):
    query: str = ""
    summary: str = ""


@node
async def research(state: Research) -> dict:
    await asyncio.sleep(0.01)
    return {"summary": f"findings on {state.query}"}


researcher = Graph(Research, name="researcher").flow(
    START >> research,
    research >> END,
)


class Report(BaseState):
    topics: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


@node
async def dispatch(state: Report, ctx: NodeContext) -> dict:
    tasks = [
        AgentTask(graph=researcher, input={"query": topic}) for topic in state.topics
    ]
    results = await ctx.gather(await ctx.spawn(tasks))
    return {"findings": [result.data["summary"] for result in results]}


graph = Graph(Report).flow(START >> dispatch, dispatch >> END)

print(graph.invoke({"topics": ["solar", "wind"]}).state.findings)
```

```text
['findings on solar', 'findings on wind']
```

`dispatch` starts one `researcher` run per topic with `ctx.spawn`, and `ctx.gather` waits for both and returns their results in order.

## Three ways

- `ctx.spawn` and `ctx.gather` start sub-agents from a node and wait for them in the same run; see [Spawn and gather](#spawn-and-gather).
- An `AgentExecutor`, such as `AsyncioExecutor`, runs sub-agents in the background, beyond the run that started them; see [Background agents](#background-agents).
- `subgraph()` makes a graph a node of another graph, which runs inside the parent's superstep and on the parent's store; see [Subgraphs](#subgraphs).

## Describe a sub-agent

An `AgentTask` describes one sub-agent for `ctx.spawn` and for an executor.

| Field | Meaning |
|---|---|
| `graph` | The graph to run |
| `input` | The input of its run |
| `name` | Optional display name |
| `state_in`, `state_out` | Optional functions that map the input and the final state |
| `on_progress` | Optional callback that receives an `AgentStatus`; see [Progress](#progress) |
| `context` | The sub-agent's own run context |

Sub-agents do not see the parent's context, so pass what they need in `context`.

## Spawn and gather

`await ctx.spawn(tasks)` starts the sub-agents and returns a handle for each. `await ctx.gather(handles)` waits for them and returns one `AgentResult` per handle, in order.

| Attribute | Meaning |
|---|---|
| `status` | `"completed"`, `"interrupted"`, `"error"` or `"cancelled"` |
| `state` | The final state, as the graph's state type |
| `data` | The final state as a dict, or what `state_out` returned |
| `error` | What stopped a failed or cancelled run |
| `thread_id` | The thread the sub-agent ran on |

Gather every sub-agent you spawn in the same run. The functions `spawn_agents` and `gather_agents` do the same outside a node.

## Progress

`AgentStatus` describes where a sub-agent is, derived from its own events. Pass `on_progress=` to receive it. This block continues the opening example:

<!-- continue -->

```python
from nodestep import AgentStatus

last_status: dict[str, str] = {}


def remember(status: AgentStatus) -> None:
    last_status[status.name] = f"{status.status} after step {status.step}"


@node
async def dispatch(state: Report, ctx: NodeContext) -> dict:
    tasks = [
        AgentTask(
            graph=researcher,
            input={"query": topic},
            name=f"research {topic}",
            on_progress=remember,
        )
        for topic in state.topics
    ]
    results = await ctx.gather(await ctx.spawn(tasks))
    return {"findings": [result.data["summary"] for result in results]}


graph = Graph(Report).flow(START >> dispatch, dispatch >> END)

graph.invoke({"topics": ["solar", "wind"]})
print(sorted(last_status.items()))
```

```text
[('research solar', 'completed after step 1'), ('research wind', 'completed after step 1')]
```

Each sub-agent ran one superstep. An `AgentStatus` has these attributes:

| Attribute | Meaning |
|---|---|
| `id`, `name` | Handle id and display name |
| `status` | `"running"`, `"completed"`, `"interrupted"`, `"error"` or `"cancelled"` |
| `step` | Supersteps the sub-agent's thread has finished |
| `node` | Last node that finished, or `None` before the first superstep |
| `thread_id` | Thread the sub-agent runs on |
| `updated_at` | `time.time()` of the last change |

`on_progress` gets it after every superstep of the sub-agent and once when it ends. `await handle.status()` returns it at any time.

## Background agents

`AsyncioExecutor` runs sub-agents as asyncio tasks in this process. They outlive the node that started them, so a later run, for example after a pause, can check on them.

```python
import asyncio

from nodestep import END, START, AgentTask, AsyncioExecutor, BaseState, Graph, node


class Job(BaseState):
    name: str = ""
    result: str = ""


@node
async def work(state: Job) -> dict:
    await asyncio.sleep(0.05)
    return {"result": f"processed {state.name}"}


worker = Graph(Job, name="worker").flow(START >> work, work >> END)
executor = AsyncioExecutor()


async def main() -> None:
    handle = await executor.submit(
        AgentTask(graph=worker, input={"name": "invoices"}), thread_prefix="jobs"
    )
    print((await executor.status(handle.id)).status)
    while (result := await executor.poll(handle.id)) is None:
        await asyncio.sleep(0.01)
    print(result.status, result.data["result"])
    print((await executor.status(handle.id)).status)
    await executor.forget(handle.id)


asyncio.run(main())
```

```text
running
completed processed invoices
completed
```

| Method | What it does |
|---|---|
| `submit(task)` | Starts the agent and returns a `DurableAgentHandle` |
| `poll(id)` | Returns the `AgentResult`, or `None` while the agent runs |
| `status(id)` | Returns its `AgentStatus` |
| `cancel(id)` | Stops it; its status becomes `"cancelled"` |
| `forget(id)` | Releases it; the executor keeps every agent it started until then |

Inside a node, `await ctx.submit(tasks, executor=executor)` submits with the run's thread id as the prefix. `await ctx.poll_agents(handles, executor=executor)` checks them without waiting.

## Subgraphs

`subgraph(name, graph, ...)` wraps a child graph as a node. Its state is wired explicitly, in one of two ways:

- `share=["messages", ...]` names fields both graphs declare: the child starts with the parent's values, and the parent gets the child's changes back.
- `state_in=` builds the child's input from the parent state, and `state_out=` builds the parent's update from the child's final state.

`child_thread=` is required and decides the child's thread. With `"fresh"`, the child starts from its defaults on every call. With `"stable"`, the child keeps its own state between calls.

```python
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Graph, add_messages, node, subgraph
from nodestep.chat import AIMessage, HumanMessage, Message


class Conversation(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)


@node
def reply(state: Conversation) -> dict:
    return {"messages": [AIMessage(content=f"You said: {state.messages[-1].content}")]}


child = Graph(Conversation, name="replier").flow(START >> reply, reply >> END)


class Session(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    turns: int = 0


@node
def count_turns(state: Session) -> dict:
    return {"turns": len(state.messages) // 2}


answer = subgraph("answer", child, share=["messages"], child_thread="fresh")
graph = Graph(Session).flow(
    START >> answer,
    answer >> count_turns,
    count_turns >> END,
)

result = graph.invoke({"messages": [HumanMessage(content="hello")]})
print([message.content for message in result.state.messages], result.state.turns)
```

```text
['hello', 'You said: hello'] 1
```

The child runs on the parent's state store and with the parent's `context`. Its interrupts pause the parent when nothing answers them.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| Context of a spawned sub-agent | Its own `context` from the `AgentTask`; spawned sub-agents do not see the parent's context |
| `ctx.gather` result order | One `AgentResult` per handle, in the order of the handles |
| A sub-agent fails or is cancelled | `gather` raises `AgentGroupError`, unless you pass `return_exceptions=True` and read each result's `status` and `error` |
| `gather(..., timeout=)` | Cancels the unfinished sub-agents and raises `AgentTimeoutError` |
| Handle ids | `gather` also accepts handle ids, so one node can spawn and keep the ids in the state, and another node of the same run can gather them |
| Thread of a spawned sub-agent | Each sub-agent runs on its own thread, `"<prefix>:sub:<handle id>"`; the prefix is `thread_prefix=`, or the parent's thread id |
| Sub-agents a run never gathered | A run that completes or pauses with spawned sub-agents it never gathered cancels them and raises `GraphExecutionError` naming them |
| A failed or stopped run | Cancels its spawned sub-agents too |
| `spawn_agents` and `gather_agents` | Do what `ctx.spawn` and `ctx.gather` do, outside a node |
| `on_progress` calls | After every superstep of the sub-agent and once when it ends |
| `on_progress` raises | The sub-agent stops with an `"error"` result holding that exception |
| Thread of a background agent | `thread_id=` of `submit`, else `"<thread_prefix>:bg:<handle id>"`, else `"bg:<handle id>"`, so two background agents never share a thread |
| Agents an executor keeps | Every agent it started, until `forget(id)` releases it |
| An unknown handle id | `AgentHandleLostError` |
| `ctx.submit` thread prefix | The run's thread id |
| Subgraph state | Wired explicitly: `share=` names fields both graphs declare, or `state_in=` and `state_out=` map the state in and out; `state_out=` returns the parent's update as a dict |
| `child_thread=` | Required; `"fresh"` or `"stable"` |
| `child_thread="fresh"` | Every call runs on a new thread, `"<parent thread>:<name>:<random hex>"`, so the child starts from its defaults each time |
| `child_thread="stable"` | Every call runs on the thread `"<parent thread>:<name>"`, so the child keeps its own state between calls, which needs a state store |
| Store and context of a child graph | The parent's state store and the parent's `context`; a child graph with its own store is refused |
| An interrupt in the child | Goes to the child's `on_interrupt` middleware first, then to the parent's; if none answers, the parent pauses, and the interrupt's key is prefixed with the subgraph task, as in `"answer/review:approve"` |

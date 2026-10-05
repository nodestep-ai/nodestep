# Run the examples

The scripts in `examples/` each show one feature in a complete graph that runs as a script and opens in the sandbox.

## When you need it

Use them to see a feature in a complete graph, or as a start for your own. Find them in [`examples/`](https://github.com/nodestep-ai/nodestep/tree/main/examples) on GitHub. Each defines a module-level `graph`, which the [sandbox](guides/sandbox.md) opens. None needs an API key; the ones with a model use `ScriptedChat`.

## Steps

1. Clone the repository and install every extra:

    ```bash
    git clone https://github.com/nodestep-ai/nodestep
    cd nodestep
    uv sync --locked --all-extras
    ```

2. Run one with `uv run --no-sync python examples/<name>.py`.

3. Or open it in the sandbox with `uv run --no-sync nodestep sandbox examples/<name>.py:graph`.

Each example below shows both commands.

## Graphs and routing

### router.py

`classify` sets the request's intent, and `branch()` routes it to `support` or `sales`. The nodes change their input in place and return it.

```bash
uv run --no-sync python examples/router.py
uv run --no-sync nodestep sandbox examples/router.py:graph
```

### send_fan_out.py

`fan_out`, declared with `@node(goto=[...])`, returns a `Command` with one `Send` per shop. The price checks run concurrently, the `add` reducer collects the offers, and `pick_best` runs once after all of them.

```bash
uv run --no-sync python examples/send_fan_out.py
uv run --no-sync nodestep sandbox examples/send_fan_out.py:graph
```

### subgraph.py

An article goes through a `proofreader` graph with its own state, used as one node with `subgraph()`. `state_in` and `state_out` map the fields, and `child_thread="fresh"` starts the child from its defaults on every call. A draft with a lowercase title and a short body is not published; the fixed version is.

```bash
uv run --no-sync python examples/subgraph.py
uv run --no-sync nodestep sandbox examples/subgraph.py:graph
```

### streaming.py

`astream` with `stream_mode=["custom", "updates", "tokens"]`: `count_rows` reports each file with `ctx.emit`, each node's update arrives as an `updates` event, and the `summarize` model node, built with `stream=True`, streams its answer token by token. The last event is `final`.

```bash
uv run --no-sync python examples/streaming.py
uv run --no-sync nodestep sandbox examples/streaming.py:graph
```

### custom_middleware.py

Two custom middleware. `RedactEmails` replaces e-mail addresses in each node's update with `ctx.replace`; `AuditLog` records `before_node`, `after_node` and `on_run_end`. `after_*` hooks run in reverse list order, so the audit log sees the redacted update.

```bash
uv run --no-sync python examples/custom_middleware.py
uv run --no-sync nodestep sandbox examples/custom_middleware.py:graph
```

## Interrupts and threads

### human_approval.py

A deployment waits for a confirmation. Shows `interrupt()` with an id, the pending interrupts of a paused result, and `Resume(True)`. In the sandbox, answer the interrupt form with `true`.

```bash
uv run --no-sync python examples/human_approval.py
uv run --no-sync nodestep sandbox examples/human_approval.py:graph
```

### ask_user.py

A greeting asks for a name. `graph` pauses until resumed with `{"name": ...}`; `prompting_graph` adds a middleware whose `on_interrupt` hook asks on the terminal, so it never pauses. In the sandbox, answer with `{"name": "Ada"}`.

```bash
uv run --no-sync python examples/ask_user.py
uv run --no-sync nodestep sandbox examples/ask_user.py:graph
```

### durable_thread.py

The first run pauses an order on `interrupt()` and exits, leaving the thread in a `FilesystemStateStore` file in the system's temporary directory. The second run resumes it from another process with the thread id the first one printed (`order-1a2b3c4d` below) and an answer.

```bash
uv run --no-sync python examples/durable_thread.py
uv run --no-sync python examples/durable_thread.py order-1a2b3c4d yes
uv run --no-sync nodestep sandbox examples/durable_thread.py:graph
```

### rewrite_message.py

A short chat thread is loaded at its first event, forked as the branch `rewrite`, and continued there with a different message, while the main branch keeps its history. Shows `history`, `load(at=...)`, `fork`, `branches` and `branch_id=`.

```bash
uv run --no-sync python examples/rewrite_message.py
uv run --no-sync nodestep sandbox examples/rewrite_message.py:graph
```

## Agents and tools

### simple_agent.py

The smallest agent: `build_react_agent` with `ScriptedChat` and no tools, asked one question with `run_agent`.

```bash
uv run --no-sync python examples/simple_agent.py
uv run --no-sync nodestep sandbox examples/simple_agent.py:graph
```

### tool_agent.py

An agent built by hand from `model_node`, `tool_runner` and `when()`, with a `LocalWorkspace` and a `ToolInterruptMiddleware` that asks before `list_files` runs. In the sandbox, approve or deny the call on the run page.

```bash
uv run --no-sync python examples/tool_agent.py
uv run --no-sync nodestep sandbox examples/tool_agent.py:graph
```

### guarded_tools.py

A `refund` tool behind two guards: `ToolLimitMiddleware(per_tool={"refund": 2})` and a `ToolInterruptMiddleware` rule with `gt("amount", 100)`. The scripted model asks for three refunds: the small one runs, a person denies the large one with `ToolDecision(action="deny", message=...)`, and the third is over the budget. Both denials reach the model as tool messages.

```bash
uv run --no-sync python examples/guarded_tools.py
uv run --no-sync nodestep sandbox examples/guarded_tools.py:graph
```

### memory_agent.py

A graph saves, lists, searches and deletes long-term memories with the tools of `FilesystemMemory`, passed as the run's `context=` with the scope `user:42` set in code. Shows `call_tool` with a `ToolContext` built from the node. In the sandbox, `--context` passes a memory in a new temporary directory.

```bash
uv run --no-sync python examples/memory_agent.py
uv run --no-sync nodestep sandbox examples/memory_agent.py:graph --context examples/memory_agent.py:temporary_memory
```

## Sub-agents

### parallel_research.py

One node starts a researcher sub-agent per topic with `ctx.spawn`, waits for all of them with `ctx.gather`, and prints each `AgentStatus` from `on_progress`.

```bash
uv run --no-sync python examples/parallel_research.py
uv run --no-sync nodestep sandbox examples/parallel_research.py:graph
```

### background_agent.py

A worker graph runs in the background on `AsyncioExecutor`, on the thread `jobs:bg:<handle id>`, while the caller checks `executor.status` and `executor.poll`. The sandbox opens the worker graph itself.

```bash
uv run --no-sync python examples/background_agent.py
uv run --no-sync nodestep sandbox examples/background_agent.py:graph
```

## The sandbox demo

The sandbox ships a support-desk agent that needs no key: a scripted model proposes a refund, and `ToolInterruptMiddleware` waits for your approval.

```bash
uv run --no-sync nodestep sandbox nodestep_sandbox.demo:graph
```

## Good to know

- The examples with a model build it in one statement, `chat = ScriptedChat(...)`. Replace it with any chat model from the [chat integrations](concepts/agents.md#chat-models); the reference page of each one shows how to create it.
- [Use the sandbox](guides/sandbox.md) explains the page the sandbox commands open.

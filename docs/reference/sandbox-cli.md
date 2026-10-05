# Sandbox CLI

The `nodestep` command comes with the `sandbox` extra; see [Install](../install.md). `nodestep sandbox` starts a local web page that runs one graph; the [sandbox guide](../guides/sandbox.md) shows what is on it.

## Usage

```text
nodestep sandbox TARGET [--context MODULE:ATTRIBUTE] [--host HOST] [--port PORT]
                        [--no-browser]
```

`TARGET` is the graph to load, written as `package.module:attribute` or `path/to/file.py:attribute`. The attribute is a `Graph`, or a function without arguments, sync or async, that returns one. The current directory is put on `sys.path`, so modules next to your code import as they do in a script.

| Option | Default | Meaning |
|---|---|---|
| `--context MODULE:ATTRIBUTE` | none | The object passed as `context=` to every run and resume, written like `TARGET`. A callable, such as a function or a class, is called once without arguments at start and its result used; an awaitable result is awaited |
| `--host HOST` | `127.0.0.1` | The one address the server binds to. Wildcard addresses (`0.0.0.0`, `::` or an empty value) are refused |
| `--port PORT` | `8765` | Port the server binds to |
| `--no-browser` | off | Do not open a browser tab on start |

Without the `sandbox` extra the command prints the install command and exits with code 2. It also prints an error and exits with code 2 when `--host` is a wildcard address, or when the graph or the context cannot be loaded: a malformed target, a missing module, file or attribute, an object that is neither a `Graph` nor a function returning one, a function that needs arguments, or a graph without a flow.

## What it runs

- The sandbox runs a shallow copy of your graph; your graph object is not changed.
- A graph without a state store gets an `InMemoryStateStore`, and the page says so. Threads then last only as long as the sandbox process.
- Runs stream with `stream_mode=["updates", "custom", "debug"]`: one row per step and task, custom events as they arrive, and at the end the final state or the interrupt form.
- The interrupt form shows the key of each pending interrupt and sends the answers as `Resume(answers=...)`. Tool approvals (`ToolInterrupt`) can be approved, edited as JSON or denied. An edit is merged into the call's arguments: the keys you give replace those of the call, and a key you delete from the JSON keeps its value.

## Security

- The server binds to `127.0.0.1` by default. `--host` takes one address; wildcard addresses are refused.
- Requests whose `Host` header is not the bound address and port are rejected.
- Every form post carries a random token created when the process starts.
- No page runs code other than the loaded graph.

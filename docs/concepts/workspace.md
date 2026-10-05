# Workspace and memory

A workspace is a set of files under one root that tools can read and change.

```python
from nodestep.workspace import InMemoryWorkspace

workspace = InMemoryWorkspace()
workspace.write_file("notes/todo.md", "- ship docs\n- fix bug\n")

print(workspace.list_dir("notes"))
print(workspace.edit("notes/todo.md", "fix bug", "fix the login bug"))
print(workspace.grep("login").matches)
```

```text
['todo.md']
1
{'notes/todo.md': ['- fix the login bug']}
```

`edit` returns how many replacements it made, and `grep` returns the matching lines by file. For long-term memory, `FilesystemMemory` gives a model four tools; see [Memory](#memory).

## Workspaces

Pass a workspace to `Graph(workspace=...)` or `build_react_agent(workspace=...)`. Tools and nodes read it as `ctx.workspace`.

| Workspace | Files |
|---|---|
| [`LocalWorkspace`](../reference/workspace/integrations/local.md) | A directory on disk |
| [`InMemoryWorkspace`](../reference/workspace/integrations/virtual.md) | Kept in memory, with the same path rules, for tests |

Both have `read_file`, `write_file`, `list_dir`, `exists`, `delete`, `edit`, `glob` and `grep`. Paths are relative to the root, and a path that could leave it raises `PathAccessError`. The rules are the same on every operating system.

```python
from nodestep import PathAccessError
from nodestep.workspace import InMemoryWorkspace

workspace = InMemoryWorkspace()

for path in ["../secrets.txt", "/etc/passwd", "notes/CON"]:
    try:
        workspace.read_file(path)
    except PathAccessError as error:
        print(error)
```

```text
Path '../secrets.txt' is outside the workspace root
Path '/etc/passwd' is absolute; use a relative path
Path 'notes/CON' uses the reserved device name 'CON'
```

The first path climbs above the root, the second is absolute, and the third is a Windows device name.

## Tools that use a workspace

A workspace has no tools of its own. Write the tools you want, and read the workspace from the `ToolContext`.

```python
import asyncio

from nodestep import ScriptedChat, ToolContext, build_react_agent, run_agent, tool
from nodestep.chat import ChatResponse, ToolCall
from nodestep.workspace import InMemoryWorkspace


@tool
def read_note(path: str, ctx: ToolContext) -> str:
    """Read a note from the workspace."""
    return ctx.workspace.read_file(path)


workspace = InMemoryWorkspace()
workspace.write_file("release.md", "Release on Friday.")

chat = ScriptedChat(
    [
        ChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1", name="read_note", arguments={"path": "release.md"}
                )
            ]
        ),
        "The release is on Friday.",
    ]
)
agent = build_react_agent(chat, tools=[read_note], workspace=workspace)

print(asyncio.run(run_agent(agent, "When is the release?")))
print(chat.requests[1].messages[-1].content)
```

```text
The release is on Friday.
{"result":"Release on Friday."}
```

The second line is the tool message the model received with its second request.

## Memory

`FilesystemMemory(root_dir, scope=...)` gives a model long-term memory as four tools:

- `save_memory` saves an entry under a key.
- `search_memory` finds entries, ranked with BM25.
- `list_memories` lists the entries.
- `delete_memory` deletes an entry.

The scope is set in code, never by the model: the tools have no scope argument. `scope=` is a string such as `"user:42"`, or a function of the call's `ToolContext`, such as `lambda ctx: f"user:{ctx.context.user_id}"`.

```python
import asyncio
import tempfile
from pathlib import Path

from nodestep import ToolContext, call_tool
from nodestep.middleware import FilesystemMemory


async def main(directory: Path) -> None:
    memory = FilesystemMemory(directory, scope="user:42")
    tools = {memory_tool.name: memory_tool for memory_tool in memory.tools()}
    print(sorted(tools))

    await call_tool(
        tools["save_memory"],
        {"key": "language", "title": "Language", "content": "Prefers Spanish"},
        ToolContext(),
    )
    found = await call_tool(tools["search_memory"], {"query": "spanish"}, ToolContext())
    print([entry.key for entry in found.value.results])
    print(
        sorted(
            path.relative_to(directory).as_posix() for path in directory.rglob("*.json")
        )
    )


with tempfile.TemporaryDirectory() as directory:
    asyncio.run(main(Path(directory)))
```

```text
['delete_memory', 'list_memories', 'save_memory', 'search_memory']
['language']
['memories/user/42/language.json']
```

A node gets the memory tools only when you list them, as in `tools=[*memory.tools(), search]`, or pass the memory to `build_react_agent(middleware=[memory])`. The `memory_agent.py` [example](../examples.md) uses these tools in a graph.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| `LocalWorkspace(root_dir, create=False)` | A missing root raises `WorkspaceError` unless `create=True` |
| Paths | Relative to the workspace root |
| A path that could leave the root | `PathAccessError`: `..` above the root, absolute paths, drive and UNC paths, NTFS alternate data streams, Windows device names such as `CON` and `NUL`, and segments that end in a dot or a space |
| Path rules on each operating system | The same everywhere, for both workspaces |
| Symlinks in a `LocalWorkspace` | Refused when they point outside the root |
| `list_dir`, `glob` and `grep` on a missing path | `FileNotFoundError` |
| `grep` result | The matching lines by file, plus the files it could not read in `skipped` |
| `edit` result | How many replacements it made; an empty `old` text is refused |
| Tools of a workspace | None; tools read the workspace as `ctx.workspace` |
| Memory entries | UTF-8 JSON files under `root_dir` |
| `search_memory` ranking | BM25 |
| Memory scope | Set in code, never by the model: the tools have no scope argument |
| `scope=` | A string such as `"user:42"`, or a function of the call's `ToolContext` |
| Where entries live | Entries of scope `"user:42"` live in `root_dir/memories/user/42/` |
| Memory tools on a node | Only when you list them, as in `tools=[*memory.tools(), search]`, or pass the memory to `build_react_agent(middleware=[memory])` |
| Memory files that cannot be read | Listed in the `skipped` field of search and list results |

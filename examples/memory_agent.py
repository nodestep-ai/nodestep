from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    NodeContext,
    ToolContext,
    call_tool,
    node,
)
from nodestep.middleware import FilesystemMemory


class MemoryState(BaseState):
    query: str = "python"
    listed: list[str] = Field(default_factory=list)
    found: list[str] = Field(default_factory=list)
    deleted: bool = False


async def use_memory(
    ctx: NodeContext, state: MemoryState, name: str, arguments: dict[str, Any]
) -> Any:
    memory: FilesystemMemory = ctx.context
    tools = {memory_tool.name: memory_tool for memory_tool in memory.tools()}
    result = await call_tool(
        tools[name], arguments, ToolContext.from_node_context(ctx, state)
    )
    return result.value


@node
async def remember(state: MemoryState, ctx: NodeContext) -> dict:
    await use_memory(
        ctx,
        state,
        "save_memory",
        {
            "key": "user-lang",
            "title": "Preferred language",
            "content": "User prefers Spanish",
        },
    )
    await use_memory(
        ctx,
        state,
        "save_memory",
        {
            "key": "py-style",
            "title": "Python style",
            "content": "Use snake_case and prefer composition over inheritance",
        },
    )
    listed = await use_memory(ctx, state, "list_memories", {})
    return {"listed": [entry.key for entry in listed.entries]}


@node
async def recall(state: MemoryState, ctx: NodeContext) -> dict:
    hits = await use_memory(ctx, state, "search_memory", {"query": state.query})
    return {"found": [hit.key for hit in hits.results]}


@node
async def forget(state: MemoryState, ctx: NodeContext) -> dict:
    deleted = await use_memory(ctx, state, "delete_memory", {"key": "py-style"})
    return {"deleted": deleted.deleted}


graph = Graph(MemoryState, name="memory").flow(
    START >> remember,
    remember >> recall,
    recall >> forget,
    forget >> END,
)


def temporary_memory() -> FilesystemMemory:
    return FilesystemMemory(Path(tempfile.mkdtemp()), scope="user:42")


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        memory = FilesystemMemory(Path(directory), scope="user:42")
        result = await graph.ainvoke({"query": "python"}, context=memory)

    print(f"listed {len(result.state.listed)} memories under user:42")
    print(f"search 'python' → {result.state.found}")
    print(f"delete py-style → deleted={result.state.deleted}")


if __name__ == "__main__":
    asyncio.run(main())

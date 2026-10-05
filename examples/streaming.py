import asyncio
from typing import Annotated

from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    NodeContext,
    ScriptedChat,
    add_messages,
    model_node,
    node,
)
from nodestep.chat import ChatStreamChunk, HumanMessage, Message, ToolCall
from nodestep.core import FinalEventData

FILE_ROWS = {"orders.csv": 120, "returns.csv": 8}


class Upload(BaseState):
    files: list[str] = Field(default_factory=lambda: list(FILE_ROWS))
    rows: int = 0
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None


@node
async def count_rows(state: Upload, ctx: NodeContext) -> dict:
    total = 0
    for name in state.files:
        await asyncio.sleep(0.01)
        rows = FILE_ROWS.get(name, 0)
        total += rows
        ctx.emit({"file": name, "rows": rows})
    question = f"Summarize the upload: {total} rows in {len(state.files)} files."
    return {"rows": total, "messages": [HumanMessage(content=question)]}


chat = ScriptedChat(default="The upload has 128 rows in 2 files.")

summarize = model_node("summarize", chat=chat, stream=True)

graph = Graph(Upload, name="upload").flow(
    START >> count_rows,
    count_rows >> summarize,
    summarize >> END,
)


async def main() -> None:
    async for event in graph.astream({}, stream_mode=["custom", "updates", "tokens"]):
        data = event.data
        if event.mode == "custom":
            print("custom", data)
        elif event.mode == "updates" and isinstance(data, dict):
            print("updates", event.node, sorted(data))
        elif isinstance(data, ChatStreamChunk) and data.content_delta:
            print("tokens", repr(data.content_delta))
        elif isinstance(data, FinalEventData):
            print("final", data.state["final_text"])


if __name__ == "__main__":
    asyncio.run(main())

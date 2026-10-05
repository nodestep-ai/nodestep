from __future__ import annotations

import asyncio

from pydantic import BaseModel

from nodestep import (
    END,
    START,
    AgentTask,
    AsyncioExecutor,
    Graph,
    node,
)


class WorkerState(BaseModel):
    task_name: str = ""
    result: str = ""


@node
async def do_work(state: WorkerState) -> dict:
    await asyncio.sleep(0.1)
    return {"result": f"completed: {state.task_name}"}


graph = Graph(WorkerState, name="worker").flow(START >> do_work, do_work >> END)

executor = AsyncioExecutor(executor_id="local")


async def main() -> None:
    handle = await executor.submit(
        AgentTask(graph=graph, input={"task_name": "data-processing"}),
        thread_prefix="jobs",
    )
    print(f"submitted: {handle.name} (id={handle.id}, thread={handle.thread_id})")

    while (result := await executor.poll(handle.id)) is None:
        status = await executor.status(handle.id)
        print(f"{status.status}: step {status.step}, last node {status.node}")
        await asyncio.sleep(0.05)
    print(f"{result.status}: {result.data}")


if __name__ == "__main__":
    asyncio.run(main())

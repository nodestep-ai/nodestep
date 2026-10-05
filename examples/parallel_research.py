from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from nodestep import END, START, AgentTask, Graph, add, node
from nodestep.core.agent import AgentStatus
from nodestep.core.stream import NodeContext


class ResearchState(BaseModel):
    query: str = ""
    summary: str = ""


@node
def research(state: ResearchState) -> dict:
    return {"summary": f"Research findings for: {state.query}"}


researcher = Graph(ResearchState, name="researcher").flow(
    START >> research,
    research >> END,
)


class OrchestratorState(BaseModel):
    topics: list[str] = Field(default_factory=list)
    findings: Annotated[list[str], add] = Field(default_factory=list)


def report(status: AgentStatus) -> None:
    print(f"  [{status.name} {status.id}] {status.status} after step {status.step}")


@node
async def dispatch_research(state: OrchestratorState, ctx: NodeContext) -> dict:
    handles = await ctx.spawn(
        [
            AgentTask(
                graph=researcher,
                input={"query": topic},
                state_out=lambda data: {"summary": data.get("summary", "")},
                on_progress=report,
            )
            for topic in state.topics
        ]
    )
    results = await ctx.gather(handles)
    return {"findings": [result.data.get("summary", "") for result in results]}


graph = Graph(OrchestratorState, name="parallel-research").flow(
    START >> dispatch_research,
    dispatch_research >> END,
)


async def main() -> None:
    result = await graph.ainvoke(
        {"topics": ["quantum computing", "gene editing", "fusion energy"]}
    )
    for finding in result.data["findings"]:
        print(f"  - {finding}")


if __name__ == "__main__":
    asyncio.run(main())

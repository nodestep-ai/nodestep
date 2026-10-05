from __future__ import annotations

from nodestep import AgentState, AgentTask, build_react_agent, run_agent
from nodestep.chat.messages import (
    ChatRequest,
    ChatResponse,
    HumanMessage,
    ToolCall,
)
from nodestep.core.agent import gather_agents, spawn_agents
from nodestep.core.graph import Graph as GraphClass
from nodestep.core.tool import tool


class MockChat:
    model = "mock-model"

    def __init__(self, responses: list[ChatResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self._call_count = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
            self._call_count += 1
            return response
        return ChatResponse(content="default response")

    async def stream(self, request: ChatRequest):  # type: ignore[override]
        raise NotImplementedError
        yield


def test_agent_returns_graph() -> None:
    chat = MockChat()
    agent = build_react_agent(chat=chat, tools=[])
    assert isinstance(agent, GraphClass)


def test_agent_has_correct_name() -> None:
    chat = MockChat()
    agent = build_react_agent(chat=chat, tools=[], name="my-agent")
    assert agent.name == "my-agent"


def test_agent_has_think_and_act_nodes() -> None:
    chat = MockChat()
    agent = build_react_agent(chat=chat, tools=[])
    assert "think" in agent.nodes
    assert "act" in agent.nodes


def test_agent_uses_agent_state_schema() -> None:
    chat = MockChat()
    agent = build_react_agent(chat=chat, tools=[])
    assert agent.state_schema.schema_type is AgentState


async def test_agent_run_returns_final_text() -> None:
    chat = MockChat(responses=[ChatResponse(content="Paris")])
    agent = build_react_agent(chat=chat, tools=[])
    answer = await run_agent(agent, "What is the capital of France?")
    assert answer == "Paris"


async def test_agent_run_with_tool_calls() -> None:
    @tool(name="add_numbers", description="Add two numbers")
    def add_numbers(a: int, b: int) -> int:
        return a + b

    chat = MockChat(
        responses=[
            ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1", name="add_numbers", arguments={"a": 2, "b": 3}
                    )
                ],
            ),
            ChatResponse(content="The answer is 5"),
        ]
    )
    agent = build_react_agent(chat=chat, tools=[add_numbers])
    answer = await run_agent(agent, "What is 2+3?")
    assert answer == "The answer is 5"


async def test_agent_max_steps_respected() -> None:
    chat = MockChat(
        responses=[
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id=f"call-{i}", name="noop", arguments={})],
            )
            for i in range(100)
        ]
    )

    @tool(name="noop", description="noop")
    def noop() -> str:
        return "ok"

    agent = build_react_agent(chat=chat, tools=[noop], max_steps=3)
    import pytest

    from nodestep.exceptions import RunLimitExceededError

    with pytest.raises(RunLimitExceededError):
        await run_agent(agent, "loop forever")


async def test_agent_tool_limit_respected() -> None:
    call_count = 0

    @tool(name="counter", description="count")
    def counter() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    chat = MockChat(
        responses=[
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id=f"call-{i}", name="counter", arguments={})],
            )
            for i in range(10)
        ]
        + [ChatResponse(content="done")]
    )

    agent = build_react_agent(
        chat=chat, tools=[counter], max_tool_calls=3, max_steps=30
    )

    answer = await run_agent(agent, "call counter many times")

    assert call_count == 3
    assert answer == "done"


async def test_agent_custom_middleware_included() -> None:
    from nodestep.middleware.base import Middleware, NodeMiddlewareContext

    hook_called = False

    class TrackingMiddleware(Middleware):
        def before_node(self, ctx: NodeMiddlewareContext) -> None:
            nonlocal hook_called
            hook_called = True

    chat = MockChat(responses=[ChatResponse(content="hi")])
    agent = build_react_agent(chat=chat, tools=[], middleware=[TrackingMiddleware()])
    await run_agent(agent, "hello")
    assert hook_called


async def test_agent_as_subagent_with_spawn_gather() -> None:
    chat = MockChat(responses=[ChatResponse(content="research done")])
    researcher = build_react_agent(chat=chat, tools=[], name="researcher")

    handles = await spawn_agents(
        [
            AgentTask(
                graph=researcher, input={"messages": [HumanMessage(content="topic")]}
            )
        ]
    )
    results = await gather_agents(handles)
    assert len(results) == 1
    assert results[0].data.get("final_text") == "research done"


async def test_agent_state_out_auto_applied_via_spawn() -> None:
    chat = MockChat(responses=[ChatResponse(content="found it")])
    researcher = build_react_agent(chat=chat, tools=[], name="researcher")

    handles = await spawn_agents(
        [
            AgentTask(
                graph=researcher,
                input={"messages": [HumanMessage(content="find")]},
                state_out=lambda data: {"summary": data.get("final_text", "")},
            )
        ]
    )
    results = await gather_agents(handles)
    assert results[0].data == {"summary": "found it"}


async def test_agent_state_in_auto_applied_via_spawn() -> None:
    chat = MockChat(responses=[ChatResponse(content="done")])
    worker = build_react_agent(chat=chat, tools=[], name="worker")

    handles = await spawn_agents(
        [
            AgentTask(
                graph=worker,
                input="raw input string",
                state_in=lambda arguments: {
                    "messages": [HumanMessage(content=str(arguments))]
                },
            )
        ]
    )
    results = await gather_agents(handles)
    assert results[0].status == "completed"


def test_agent_mermaid_renders() -> None:
    chat = MockChat()
    agent = build_react_agent(chat=chat, tools=[])
    mermaid = agent.to_mermaid()
    assert "think" in mermaid
    assert "act" in mermaid


async def test_agent_with_output_schema() -> None:
    import json

    from pydantic import BaseModel as PydanticBaseModel

    class Answer(PydanticBaseModel):
        text: str

    answer_json = json.dumps({"text": "hello"})

    async def mock_complete(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content=answer_json)

    async def mock_stream(self, request):
        raise NotImplementedError
        yield

    class FakeChat:
        model = "fake-model"
        complete = mock_complete
        stream = mock_stream

    chat = FakeChat()
    agent = build_react_agent(chat, tools=[], output_schema=Answer)
    result = await agent.ainvoke({"messages": [HumanMessage(content="test")]})
    assert result.data.get("final_output") is not None
    assert result.data["final_output"].text == "hello"


async def test_run_agent_raises_when_the_run_is_interrupted() -> None:
    import pytest

    from nodestep import InMemoryStateStore
    from nodestep.exceptions import RunInterruptedError
    from nodestep.middleware import InterruptRule, ToolInterruptMiddleware

    @tool(name="deploy", description="deploy")
    def deploy() -> str:
        return "ok"

    chat = MockChat(
        responses=[
            ChatResponse(tool_calls=[ToolCall(id="d", name="deploy", arguments={})])
        ]
    )
    agent = build_react_agent(
        chat,
        tools=[deploy],
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="deploy")])],
        state_store=InMemoryStateStore(),
    )

    with pytest.raises(RunInterruptedError) as info:
        await run_agent(agent, "ship it", thread_id="t")

    assert list(info.value.interrupts) == ["act:approve_d"]
    assert info.value.thread_id == "t"

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import BaseModel, Field

from nodestep import END, START, Graph, add_messages, node, tool
from nodestep.chat import Message, ToolCall
from nodestep.core import tool_runner
from nodestep.core.tool import ToolContext, call_tool
from nodestep.middleware import (
    FilesystemSkills,
    LoadedSkill,
    Middleware,
    Replacement,
    SkillCatalog,
    SkillNotFoundError,
    TodoItem,
    TodoItemState,
    TodoListMiddleware,
    TodoListMiddlewareError,
    TodoListSnapshot,
    collect_middleware_tools,
    run_after_hooks,
    run_before_hooks,
)


@dataclass(frozen=True, slots=True)
class HookContext:
    value: Any

    def replace(self, value: Any) -> Replacement:
        return Replacement(value)


class First(Middleware):
    def before_node(self, ctx):
        return ctx.replace([*ctx.value, "first-before"])

    def after_node(self, ctx):
        return ctx.replace([*ctx.value, "first-after"])


class Second(Middleware):
    def before_node(self, ctx):
        return ctx.replace([*ctx.value, "second-before"])

    def after_node(self, ctx):
        return ctx.replace([*ctx.value, "second-after"])


@pytest.mark.asyncio
async def test_before_hooks_run_in_middleware_order() -> None:
    async def make_ctx(value: Any) -> HookContext:
        return HookContext(value)

    result = await run_before_hooks(
        [First(), Second()],
        "before_node",
        ["start"],
        make_ctx,
    )

    assert result == ["start", "first-before", "second-before"]


@pytest.mark.asyncio
async def test_after_hooks_run_in_reverse_middleware_order() -> None:
    async def make_ctx(value: Any) -> HookContext:
        return HookContext(value)

    result = await run_after_hooks(
        [First(), Second()],
        "after_node",
        ["start"],
        make_ctx,
    )

    assert result == ["start", "second-after", "first-after"]


@pytest.mark.asyncio
async def test_none_keeps_current_value() -> None:
    class Observer(Middleware):
        def before_node(self, ctx):
            return None

    async def make_ctx(value: Any) -> HookContext:
        return HookContext(value)

    result = await run_before_hooks([Observer()], "before_node", {"x": None}, make_ctx)

    assert result == {"x": None}


@pytest.mark.asyncio
async def test_replacement_can_replace_with_none() -> None:
    class Nuller(Middleware):
        def before_node(self, ctx):
            return ctx.replace(None)

    async def make_ctx(value: Any) -> HookContext:
        return HookContext(value)

    result = await run_before_hooks([Nuller()], "before_node", {"x": 1}, make_ctx)

    assert result is None


@node
def add_one(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


@node
def finish(state: dict) -> dict:
    return {"finished": True, "count": state["count"]}


def test_graph_middleware_observes_initial_and_final_state() -> None:
    observed: list[tuple[str, dict]] = []

    class GraphStateMiddleware(Middleware):
        def before_graph(self, ctx):
            observed.append(("before_graph", ctx.state))

        def after_graph(self, ctx):
            observed.append(("after_graph", ctx.state))

    graph = Graph(dict, middleware=[GraphStateMiddleware()]).flow(
        START >> add_one, add_one >> END
    )

    result = graph.invoke({"count": 1})

    assert observed == [("before_graph", {"count": 1}), ("after_graph", {"count": 2})]
    assert result.data == {"count": 2}


def test_node_middleware_transforms_input_and_output() -> None:
    class NodeStateMiddleware(Middleware):
        def before_node(self, ctx):
            return ctx.replace({"count": ctx.state.get("count", 0) + 100})

        def after_node(self, ctx):
            return ctx.replace({**ctx.state, "node_seen": ctx.node_name})

    graph = Graph(dict, middleware=[NodeStateMiddleware()]).flow(
        START >> add_one, add_one >> END
    )

    result = graph.invoke({"count": 1})

    assert result.data == {"count": 102, "node_seen": "add_one"}


def test_node_middleware_can_block_execution() -> None:
    class BlockingMiddleware(Middleware):
        def before_node(self, ctx):
            raise PermissionError(f"blocked {ctx.node_name}")

    graph = Graph(dict, middleware=[BlockingMiddleware()]).flow(
        START >> add_one, add_one >> END
    )

    with pytest.raises(PermissionError, match="blocked add_one"):
        graph.invoke({"count": 1})


class NumberResult(BaseModel):
    value: int


@tool
async def double(value: int) -> NumberResult:
    """Double."""
    return NumberResult(value=value * 2)


class ToolState(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


@node
def request_double(_: dict) -> dict:
    return {
        "tool_calls": [
            ToolCall(id="call_1", name="double", arguments={"value": 3}),
        ]
    }


def test_tool_middleware_transforms_args_and_result() -> None:
    class ToolStateMiddleware(Middleware):
        def before_tool(self, ctx):
            return ctx.replace({"value": ctx.value.value + 4})

        def after_tool(self, ctx):
            return ctx.replace(NumberResult(value=ctx.value.value + 1))

    runner = tool_runner("tools", tools=[double])
    graph = Graph(ToolState, middleware=[ToolStateMiddleware()]).flow(
        START >> request_double,
        request_double >> runner,
        runner >> END,
    )

    result = graph.invoke({"messages": []})

    assert result.data["messages"][0].content == '{"value":15}'


def test_tool_middleware_can_block_tool_call() -> None:
    class BlockingToolMiddleware(Middleware):
        def before_tool(self, ctx):
            raise PermissionError(f"blocked {ctx.tool_name}")

    runner = tool_runner("tools", tools=[double])
    graph = Graph(ToolState, middleware=[BlockingToolMiddleware()]).flow(
        START >> request_double,
        request_double >> runner,
        runner >> END,
    )

    with pytest.raises(PermissionError, match="blocked double"):
        graph.invoke({"messages": []})


@tool(name="middleware_tool")
async def middleware_tool() -> NumberResult:
    """Middleware tool."""
    return NumberResult(value=7)


class ToolProvidingMiddleware(Middleware):
    def tools(self):
        return [middleware_tool]


def test_collect_middleware_tools_returns_tools_in_order() -> None:
    assert collect_middleware_tools([ToolProvidingMiddleware()]) == [middleware_tool]


@node
def request_middleware_tool(_: dict) -> dict:
    return {
        "tool_calls": [
            ToolCall(id="call_1", name="middleware_tool", arguments={}),
        ]
    }


def test_tool_runner_runs_middleware_tools_spliced_into_its_tools() -> None:
    provider = ToolProvidingMiddleware()
    runner = tool_runner("tools", tools=[*provider.tools()])
    graph = Graph(ToolState, middleware=[provider]).flow(
        START >> request_middleware_tool,
        request_middleware_tool >> runner,
        runner >> END,
    )

    result = graph.invoke({"messages": []})

    assert result.data["messages"][0].content == '{"value":7}'


def _skill(folder, name: str, description: str, body: str = "# Body\n") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


def test_skills_middleware_catalog_reads_skill_md(tmp_path) -> None:
    _skill(tmp_path / "research", "research", "Do research things.")

    catalog = SkillCatalog([tmp_path])
    skills = catalog.list_skills()

    assert len(skills) == 1
    assert skills[0].name == "research"
    assert skills[0].description == "Do research things."


@pytest.mark.asyncio
async def test_skills_middleware_provides_load_skill_tool(tmp_path) -> None:
    _skill(tmp_path / "research", "research", "Research", "# Research\n\nbody")

    middleware = FilesystemSkills([tmp_path], nodes={"think"})
    [load_skill] = middleware.tools()

    raw: Any = (await call_tool(load_skill, {"name": "research"})).value
    result: LoadedSkill = raw
    assert result.content.endswith("# Research\n\nbody")


@pytest.mark.asyncio
async def test_skills_middleware_missing_skill_raises(tmp_path) -> None:
    _skill(tmp_path / "research", "research", "Dig in.")
    middleware = FilesystemSkills([tmp_path], nodes={"think"})
    [load_skill] = middleware.tools()

    with pytest.raises(SkillNotFoundError):
        await call_tool(load_skill, {"name": "missing"})


@pytest.mark.asyncio
async def test_skills_middleware_load_skill_returns_directory_location(
    tmp_path,
) -> None:
    skill_dir = tmp_path / "research"
    _skill(skill_dir, "research", "Research")
    (skill_dir / "EXAMPLES.md").write_text("example", encoding="utf-8")

    middleware = FilesystemSkills([tmp_path], nodes={"think"})
    [load_skill] = middleware.tools()

    raw: Any = (
        await call_tool(
            load_skill,
            {"name": "research"},
            ToolContext(middleware=(middleware,)),
        )
    ).value

    assert raw.path.endswith("research")
    assert Path(raw.primary_file).parts[-2:] == ("research", "SKILL.md")


@pytest.mark.asyncio
async def test_todolist_middleware_tools_share_state() -> None:
    middleware = TodoListMiddleware()
    read_todos, write_todos = middleware.tools()
    ctx = ToolContext(thread_id="t")

    initial_raw: Any = (await call_tool(read_todos, {}, ctx)).value
    initial: TodoListSnapshot = initial_raw
    assert initial.items == []

    after_write_raw: Any = (
        await call_tool(
            write_todos,
            {"items": [{"task": "task one", "state": "pending"}]},
            ctx,
        )
    ).value
    after_write: TodoListSnapshot = after_write_raw
    assert after_write.items == [TodoItem(task="task one", state=TodoItemState.PENDING)]

    after_read_raw: Any = (await call_tool(read_todos, {}, ctx)).value
    after_read: TodoListSnapshot = after_read_raw
    assert after_read.items == [TodoItem(task="task one", state=TodoItemState.PENDING)]

    extended_raw: Any = (
        await call_tool(
            write_todos,
            {
                "items": [{"task": "task two", "state": "in_progress"}],
                "replace": False,
            },
            ctx,
        )
    ).value
    extended: TodoListSnapshot = extended_raw
    assert extended.items == [
        TodoItem(task="task one", state=TodoItemState.PENDING),
        TodoItem(task="task two", state=TodoItemState.IN_PROGRESS),
    ]


@pytest.mark.asyncio
async def test_todolist_supports_all_six_states() -> None:
    middleware = TodoListMiddleware()
    _, write_todos = middleware.tools()
    ctx = ToolContext(thread_id="t")

    items = [
        {"task": "a", "state": "pending"},
        {"task": "b", "state": "in_progress"},
        {"task": "c", "state": "completed"},
        {"task": "d", "state": "blocked"},
        {"task": "e", "state": "cancelled"},
        {"task": "f", "state": "failed"},
    ]
    raw: Any = (await call_tool(write_todos, {"items": items}, ctx)).value
    snapshot: TodoListSnapshot = raw
    assert [item.state for item in snapshot.items] == [
        TodoItemState.PENDING,
        TodoItemState.IN_PROGRESS,
        TodoItemState.COMPLETED,
        TodoItemState.BLOCKED,
        TodoItemState.CANCELLED,
        TodoItemState.FAILED,
    ]


@pytest.mark.asyncio
async def test_todolist_invalid_state_raises() -> None:
    from pydantic import ValidationError

    middleware = TodoListMiddleware()
    _, write_todos = middleware.tools()
    ctx = ToolContext(thread_id="t")

    with pytest.raises(ValidationError):
        await call_tool(
            write_todos,
            {"items": [{"task": "x", "state": "nope"}]},
            ctx,
        )


@pytest.mark.asyncio
async def test_todolist_tool_without_a_thread_raises() -> None:
    middleware = TodoListMiddleware()
    read_todos, _ = middleware.tools()

    with pytest.raises(TodoListMiddlewareError, match="thread"):
        await call_tool(read_todos, {}, ToolContext())


async def test_todo_tools_are_bound_to_their_middleware_instance() -> None:
    first, second = TodoListMiddleware(), TodoListMiddleware()
    _, write_first = first.tools()
    read_second, _ = second.tools()
    both = ToolContext(middleware=(second, first), thread_id="t")

    await call_tool(write_first, {"items": [{"task": "only in first"}]}, both)
    seen: Any = (await call_tool(read_second, {}, both)).value

    assert seen.items == []
    assert [item.task for item in first.items_for("t")] == ["only in first"]
    assert first.tools()[0] is not second.tools()[0]


async def test_todo_tools_work_in_a_graph_without_the_middleware_installed() -> None:
    from nodestep import ScriptedChat, build_react_agent
    from nodestep.chat import ChatResponse, HumanMessage

    todos = TodoListMiddleware()
    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_todos",
                        arguments={"items": [{"task": "plan"}]},
                    )
                ]
            ),
            ChatResponse(content="done"),
        ]
    )
    agent = build_react_agent(chat, tools=[*todos.tools()])

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="go")]}, thread_id="t"
    )

    assert result.data["final_text"] == "done"
    assert [item.task for item in todos.items_for("t")] == ["plan"]


def test_skills_middleware_provides_tools(tmp_path) -> None:
    _skill(tmp_path / "skills" / "code-reviewer", "code-reviewer", "Review code.")
    skills_middleware = FilesystemSkills([tmp_path / "skills"], nodes={"think"})

    tool_names = {tool.name for tool in collect_middleware_tools((skills_middleware,))}

    assert tool_names == {"load_skill"}


def test_after_node_runs_on_an_on_error_replacement() -> None:
    seen: list[Any] = []

    @node
    def explode(state: dict) -> dict:
        raise RuntimeError("boom")

    class Recover(Middleware):
        def on_error(self, ctx, error):
            return {"recovered": True}

        def after_node(self, ctx):
            seen.append(ctx.state)
            return ctx.replace({**ctx.state, "after": True})

    graph = Graph(dict, middleware=[Recover()]).flow(START >> explode, explode >> END)

    result = graph.invoke({})

    assert seen == [{"recovered": True}]
    assert result.data == {"recovered": True, "after": True}


def test_view_injected_by_before_node_is_not_persisted() -> None:
    from typing import Annotated

    from nodestep import BaseState, add

    class Chat(BaseState):
        messages: Annotated[list[str], add] = Field(default_factory=list)

    @node
    def reply(state: Chat) -> Chat:
        state.messages = [*state.messages, "reply"]
        return state

    class Inject(Middleware):
        def before_node(self, ctx):
            return ctx.replace(
                ctx.state.model_copy(
                    update={"messages": ["INJECTED", *ctx.state.messages]}
                )
            )

    graph = Graph(Chat, middleware=[Inject()]).flow(START >> reply, reply >> END)

    result = graph.invoke({"messages": ["hi"]})

    assert result.data["messages"] == ["hi", "reply"]


def test_node_hooks_of_one_task_share_a_scratch_dict() -> None:
    seen: list[tuple[str, Any]] = []

    @node
    def passthrough(state: dict) -> dict:
        return {}

    class Scratch(Middleware):
        def before_node(self, ctx):
            ctx.scratch["started"] = ctx.task_id

        def after_node(self, ctx):
            seen.append((ctx.task_id, ctx.scratch.get("started")))

    graph = Graph(dict, middleware=[Scratch()]).flow(
        START >> passthrough, passthrough >> END
    )

    graph.invoke({})

    assert seen == [("passthrough", "passthrough")]


def test_node_hooks_see_the_stored_input_as_stored_state() -> None:
    seen: list[tuple[str, Any, Any]] = []

    @node
    def passthrough(state: dict) -> dict:
        return {"answer": 42}

    class Inject(Middleware):
        def before_node(self, ctx):
            return ctx.replace({**ctx.state, "view": True})

    class Record(Middleware):
        def before_node(self, ctx):
            seen.append(("before_node", ctx.state, dict(ctx.stored_state)))
            ctx.stored_state["mutated"] = True

        def after_node(self, ctx):
            seen.append(("after_node", ctx.state, ctx.stored_state))

    graph = Graph(dict, middleware=[Inject(), Record()]).flow(
        START >> passthrough, passthrough >> END
    )

    result = graph.invoke({"question": "q"})

    assert seen == [
        ("before_node", {"question": "q", "view": True}, {"question": "q"}),
        ("after_node", {"answer": 42}, {"question": "q"}),
    ]
    assert result.data == {"question": "q", "answer": 42}


async def test_tool_limit_budget_is_per_run_under_concurrency() -> None:
    import asyncio

    from nodestep import ScriptedChat, build_react_agent
    from nodestep.chat import ChatResponse, HumanMessage, ToolMessage

    @tool
    def ping(n: int) -> int:
        """Ping."""
        return n

    def script() -> list[ChatResponse]:
        return [
            ChatResponse(
                tool_calls=[
                    ToolCall(id=f"p{i}", name="ping", arguments={"n": i})
                    for i in range(3)
                ]
            ),
            ChatResponse(content="done"),
        ]

    class TwoRuns:
        model = "scripted"

        def __init__(self) -> None:
            self.chats = {"a": ScriptedChat(script()), "b": ScriptedChat(script())}

        async def complete(self, request):
            key = "a" if request.messages[0].content == "a" else "b"
            await asyncio.sleep(0.01)
            return await self.chats[key].complete(request)

        def stream(self, request):
            raise NotImplementedError

    agent = build_react_agent(TwoRuns(), tools=[ping], max_tool_calls=3)

    first, second = await asyncio.gather(
        agent.ainvoke({"messages": [HumanMessage(content="a")]}),
        agent.ainvoke({"messages": [HumanMessage(content="b")]}),
    )

    for result in (first, second):
        replies = [
            message.content
            for message in result.data["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert not any("limit" in (reply or "") for reply in replies)


async def test_tool_limit_budget_survives_resumes() -> None:
    from nodestep import InMemoryStateStore, Resume, ScriptedChat, build_react_agent
    from nodestep.chat import ChatResponse, HumanMessage, ToolMessage
    from nodestep.middleware import InterruptRule, ToolInterruptMiddleware

    transfers: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Transfer."""
        transfers.append(amount)
        return "ok"

    chat = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id=f"t{i}", name="transfer", arguments={"amount": i})
                ]
            )
            for i in range(3)
        ]
        + [ChatResponse(content="done")]
    )
    agent = build_react_agent(
        chat,
        tools=[transfer],
        max_tool_calls=2,
        middleware=[ToolInterruptMiddleware(rules=[InterruptRule(tool="transfer")])],
        state_store=InMemoryStateStore(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="pay")]}, thread_id="t"
    )
    while result.status == "interrupted":
        result = await agent.ainvoke(resume=Resume(True), thread_id="t")

    assert transfers == [0, 1]
    replies = [
        message.content
        for message in result.data["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert "limit" in (replies[-1] or "")


async def test_todos_are_kept_per_thread() -> None:
    middleware = TodoListMiddleware()
    read_todos, write_todos = middleware.tools()
    alice = ToolContext(thread_id="alice")
    bob = ToolContext(thread_id="bob")

    await call_tool(write_todos, {"items": [{"task": "wire $5000"}]}, alice)
    bobs: Any = (await call_tool(read_todos, {}, bob)).value
    alices: Any = (await call_tool(read_todos, {}, alice)).value

    assert bobs.items == []
    assert [item.task for item in alices.items] == ["wire $5000"]


def test_same_skill_name_in_two_roots_is_a_conflict(tmp_path) -> None:
    from nodestep.middleware.skills import SkillConflictError

    for team in ("team_a", "team_b"):
        _skill(tmp_path / team / "research", "research", f"Research ({team})")

    with pytest.raises(SkillConflictError, match="research"):
        SkillCatalog([tmp_path / "team_a", tmp_path / "team_b"])


def test_listing_one_root_twice_is_not_a_conflict(tmp_path) -> None:
    _skill(tmp_path / "research", "research", "Research")

    catalog = SkillCatalog([tmp_path, tmp_path])

    assert [skill.name for skill in catalog.list_skills()] == ["research"]

import gc
from typing import Annotated, Any, TypedDict

import pytest
from pydantic import Field

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    InMemoryStateStore,
    ScriptedChat,
    node,
    tool_runner,
)
from nodestep import tool as _tool
from nodestep.chat import (
    AIMessage,
    BaseMessage,
    ChatRequest,
    ChatResponse,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from nodestep.core import model_node
from nodestep.core.builtin_nodes.agent import AgentState
from nodestep.core.command import interrupt
from nodestep.core.tool import ToolContext, call_tool
from nodestep.exceptions import GraphConfigError, ModelProviderError
from nodestep.middleware import (
    ApproximateTokenCounter,
    FilesystemSkills,
    FixedTokenTrigger,
    MessageCountTrigger,
    Middleware,
    RecentMessagesPolicy,
    RecentTokensPolicy,
    SummarizationMiddleware,
    SummarizationResult,
    TokenLimitTrigger,
)
from nodestep.utils.reducers import add_messages

Convo = AgentState


class CountingChat:
    model = "counting-model"

    def __init__(self, text: str | None = "SUMMARY") -> None:
        self.text = text
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(content=self.text)

    async def stream(self, request: ChatRequest):  # type: ignore[override]
        raise NotImplementedError
        yield


def _summarizer(
    chat: Any, *, max_messages: int, keep_recent: int
) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        chat,
        trigger=MessageCountTrigger(max_messages=max_messages),
        recent=RecentMessagesPolicy(keep_recent=keep_recent),
    )


def _humans(count: int) -> list[BaseMessage]:
    return [HumanMessage(content=f"m{i}") for i in range(count)]


def _contents(messages: list[Any]) -> list[str | None]:
    return [message.content for message in messages]


def _one_model_graph(middleware: list[Middleware], chat: Any, **kwargs: Any) -> Graph:
    think = model_node("think", chat=chat, **kwargs)
    return Graph(AgentState, middleware=middleware).flow(START >> think, think >> END)


async def test_the_request_is_compressed_when_the_trigger_fires() -> None:
    summarizer = CountingChat()
    model = ScriptedChat(["reply"])
    graph = _one_model_graph(
        [_summarizer(summarizer, max_messages=10, keep_recent=3)], model
    )

    result = await graph.ainvoke({"messages": _humans(15)})

    sent = model.requests[0].messages
    assert isinstance(sent[0], SystemMessage)
    assert _contents(sent) == ["SUMMARY", "m12", "m13", "m14"]
    assert _contents(result.data["messages"]) == [
        "SUMMARY",
        "m12",
        "m13",
        "m14",
        "reply",
    ]
    assert len(summarizer.requests) == 1


async def test_nothing_happens_below_the_trigger() -> None:
    summarizer = CountingChat()
    model = ScriptedChat(["reply"])
    graph = _one_model_graph(
        [_summarizer(summarizer, max_messages=10, keep_recent=3)], model
    )

    result = await graph.ainvoke({"messages": _humans(5)})

    assert _contents(model.requests[0].messages) == [f"m{i}" for i in range(5)]
    assert _contents(result.data["messages"]) == [*(f"m{i}" for i in range(5)), "reply"]
    assert summarizer.requests == []


async def test_the_summary_prompt_and_transcript_reach_the_summarizer() -> None:
    summarizer = CountingChat()
    middleware = SummarizationMiddleware(
        summarizer,
        trigger=MessageCountTrigger(max_messages=2),
        recent=RecentMessagesPolicy(keep_recent=1),
        summary_prompt="My custom summary prompt.",
    )

    await _one_model_graph([middleware], ScriptedChat(["ok"])).ainvoke(
        {"messages": _humans(5)}
    )

    request = summarizer.requests[0]
    assert request.messages[0] == SystemMessage(content="My custom summary prompt.")
    transcript = request.messages[1].content or ""
    assert transcript.splitlines() == [f"user: m{i}" for i in range(4)]


async def test_the_system_prompt_stays_first_and_is_not_stored() -> None:
    summarizer = CountingChat()
    model = ScriptedChat(["reply"])
    graph = _one_model_graph(
        [_summarizer(summarizer, max_messages=3, keep_recent=1)],
        model,
        system_prompt="SYS",
    )

    result = await graph.ainvoke({"messages": _humans(4)})

    assert _contents(model.requests[0].messages) == ["SYS", "SUMMARY", "m3"]
    assert "SYS" not in (summarizer.requests[0].messages[1].content or "")
    assert _contents(result.data["messages"]) == ["SUMMARY", "m3", "reply"]


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (
            ["h0", "h1", "NOTE", "h2", "h3"],
            ["SUMMARY", "h2", "h3"],
        ),
        (
            ["h0", "h1", "h2", "NOTE", "h3"],
            ["SUMMARY", "NOTE", "h3"],
        ),
    ],
)
async def test_system_messages_are_never_moved(
    history: list[str], expected: list[str]
) -> None:
    model = ScriptedChat(["reply"])
    graph = _one_model_graph(
        [_summarizer(CountingChat(), max_messages=3, keep_recent=2)], model
    )
    messages = [
        SystemMessage(content=text) if text == "NOTE" else HumanMessage(content=text)
        for text in history
    ]

    result = await graph.ainvoke({"messages": messages})

    assert _contents(model.requests[0].messages) == expected
    assert _contents(result.data["messages"]) == [*expected, "reply"]


@pytest.mark.parametrize("text", ["", None])
async def test_an_empty_summary_raises_before_the_model_is_called(
    text: str | None,
) -> None:
    model = ScriptedChat(["reply"])
    store = InMemoryStateStore()
    think = model_node("think", chat=model)
    graph = Graph(
        AgentState,
        middleware=[_summarizer(CountingChat(text), max_messages=3, keep_recent=1)],
        state_store=store,
    ).flow(START >> think, think >> END)

    with pytest.raises(ModelProviderError, match="empty summary"):
        await graph.ainvoke({"messages": _humans(5)}, thread_id="t")

    assert model.requests == []
    assert _contents((await graph.load("t"))["messages"]) == [f"m{i}" for i in range(5)]


async def test_summarization_runs_only_before_model_calls() -> None:
    seen: list[list[Any]] = []

    @node
    def prepare(state: AgentState) -> dict:
        seen.append(list(state.messages))
        return {}

    summarizer = CountingChat()
    think = model_node("think", chat=ScriptedChat(["reply"]))
    graph = Graph(
        AgentState, middleware=[_summarizer(summarizer, max_messages=3, keep_recent=1)]
    ).flow(START >> prepare, prepare >> think, think >> END)

    await graph.ainvoke({"messages": _humans(6)})

    assert _contents(seen[0]) == [f"m{i}" for i in range(6)]
    assert len(summarizer.requests) == 1


class InjectCatalog(Middleware):
    def before_node(self, ctx):
        return ctx.replace(
            ctx.state.model_copy(
                update={
                    "messages": [SystemMessage(content="VIEW"), *ctx.state.messages]
                }
            )
        )


@pytest.mark.parametrize("summary_first", [True, False])
async def test_view_only_messages_of_other_middleware_are_never_stored(
    tmp_path: Any, summary_first: bool
) -> None:
    root = tmp_path / "skills"
    (root / "research").mkdir(parents=True)
    (root / "research" / "SKILL.md").write_text(
        "---\nname: research\ndescription: Dig in\n---\n", encoding="utf-8"
    )
    summarization = _summarizer(CountingChat(), max_messages=3, keep_recent=1)
    others: list[Middleware] = [
        InjectCatalog(),
        FilesystemSkills([root], nodes={"think"}),
    ]
    middleware = [summarization, *others] if summary_first else [*others, summarization]
    model = ScriptedChat(["reply"])

    result = await _one_model_graph(middleware, model).ainvoke({"messages": _humans(5)})

    stored = _contents(result.data["messages"])
    assert stored == ["SUMMARY", "m4", "reply"]
    sent = _contents(model.requests[0].messages)
    assert "SUMMARY" in sent
    assert sent[-1] == "m4"


class InjectWithId(Middleware):
    def __init__(self, position: int) -> None:
        self.position = position

    def before_node(self, ctx):
        messages = list(ctx.state.messages)
        messages.insert(self.position, SystemMessage(content="VIEW", id="view-1"))
        return ctx.replace(ctx.state.model_copy(update={"messages": messages}))


@pytest.mark.parametrize("position", [0, 2])
async def test_view_only_messages_with_an_id_are_never_stored(position: int) -> None:
    summarization = _summarizer(CountingChat(), max_messages=3, keep_recent=1)
    model = ScriptedChat(["reply"])
    graph = _one_model_graph([InjectWithId(position), summarization], model)

    result = await graph.ainvoke({"messages": _humans(5)})

    stored = result.data["messages"]
    assert _contents(stored) == ["SUMMARY", "m4", "reply"]
    assert "view-1" not in {message.id for message in stored}


async def test_summary_is_persisted_and_not_recomputed_by_the_next_node() -> None:
    first = model_node("first", chat=ScriptedChat(["first"]))
    second = model_node("second", chat=ScriptedChat(["second"]))
    summarizer = CountingChat()
    graph = Graph(
        AgentState,
        middleware=[_summarizer(summarizer, max_messages=6, keep_recent=2)],
        state_store=InMemoryStateStore(),
    ).flow(START >> first, first >> second, second >> END)

    result = await graph.ainvoke({"messages": _humans(10)}, thread_id="s")

    expected = ["SUMMARY", "m8", "m9", "first", "second"]
    assert len(summarizer.requests) == 1
    assert _contents(result.data["messages"]) == expected
    assert _contents((await graph.load("s"))["messages"]) == expected


async def test_summarizer_input_stays_bounded_in_a_tool_loop() -> None:
    from nodestep import build_react_agent, tool

    @tool
    def step(n: int) -> int:
        """Step."""
        return n

    model = ScriptedChat(
        [
            ChatResponse(
                tool_calls=[ToolCall(id=f"s{i}", name="step", arguments={"n": i})]
            )
            for i in range(12)
        ]
        + [ChatResponse(content="done")]
    )
    summarizer = CountingChat()
    agent = build_react_agent(
        model,
        tools=[step],
        max_steps=40,
        middleware=[_summarizer(summarizer, max_messages=8, keep_recent=4)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

    transcript_sizes = [
        len((request.messages[1].content or "").splitlines())
        for request in summarizer.requests
    ]
    assert result.data["final_text"] == "done"
    assert max(transcript_sizes) <= 9
    assert len(summarizer.requests) < 12
    assert _pairs_are_intact(result.data["messages"])
    for request in model.requests:
        assert _pairs_are_intact(list(request.messages))


async def test_summarizer_request_contains_no_tool_messages() -> None:
    summarizer = CountingChat()

    await _one_model_graph(
        [_summarizer(summarizer, max_messages=2, keep_recent=1)],
        ScriptedChat(["ok"]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content=None, tool_calls=[ToolCall(id="t", name="look")]),
                ToolMessage(content="r", name="look", tool_call_id="t"),
                HumanMessage(content="next"),
            ]
        }
    )

    sent = summarizer.requests[0].messages
    assert [type(message).__name__ for message in sent] == [
        "SystemMessage",
        "HumanMessage",
    ]
    assert "look" in (sent[1].content or "")


@_tool
def look() -> str:
    """Look."""
    return "seen"


async def test_tool_nodes_are_not_summarized_and_pairs_stay_intact() -> None:
    summarizer = CountingChat()
    act = tool_runner("act", tools=[look])
    think = model_node("think", chat=ScriptedChat(["done"]))
    graph = Graph(
        AgentState,
        middleware=[_summarizer(summarizer, max_messages=1, keep_recent=0)],
    ).flow(START >> act, act >> think, think >> END)
    call = ToolCall(id="c1", name="look")

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content=None, tool_calls=[call]),
            ],
            "tool_calls": [call],
        }
    )

    assert len(summarizer.requests) == 1
    assert "tool look" in (summarizer.requests[0].messages[1].content or "")
    assert _contents(result.data["messages"]) == ["SUMMARY", "done"]


async def test_a_pending_tool_call_is_kept_with_its_result() -> None:
    model = ScriptedChat(["ok"])
    graph = _one_model_graph(
        [_summarizer(CountingChat(), max_messages=3, keep_recent=2)], model
    )
    reused = ToolCall(id="call_0", name="look")

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="one"),
                AIMessage(content=None, tool_calls=[reused]),
                ToolMessage(content="r1", name="look", tool_call_id="call_0"),
                HumanMessage(content="two"),
                AIMessage(content=None, tool_calls=[reused]),
                ToolMessage(content="r2", name="look", tool_call_id="call_0"),
            ]
        }
    )

    sent = list(model.requests[0].messages)
    assert [type(message).__name__ for message in sent] == [
        "SystemMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert _pairs_are_intact(sent)


class FanState(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None
    left_calls: list[ToolCall] = Field(default_factory=list)
    left_text: str | None = None
    right_calls: list[ToolCall] = Field(default_factory=list)
    right_text: str | None = None


async def test_parallel_model_nodes_are_not_summarized() -> None:
    def side(name: str) -> Any:
        return model_node(
            name,
            chat=ScriptedChat([name]),
            tool_calls_field=f"{name}_calls",
            final_text_field=f"{name}_text",
        )

    from nodestep import Command, Send

    left, right = side("left"), side("right")

    @node(goto=[left, right])
    def begin(state: FanState) -> Command:
        return Command(goto=[Send(left, {}), Send(right, {})])

    join = model_node("join", chat=ScriptedChat(["join"]))
    summarizer = CountingChat()
    graph = Graph(
        FanState, middleware=[_summarizer(summarizer, max_messages=4, keep_recent=2)]
    ).flow(
        START >> begin,
        left >> join,
        right >> join,
        join >> END,
    )

    result = await graph.ainvoke({"messages": _humans(5)})

    contents = _contents(result.data["messages"])
    assert len(summarizer.requests) == 1
    assert contents[0] == "SUMMARY"
    assert contents[-1] == "join"
    assert {"left", "right"} <= set(contents)


def test_trigger_and_recent_are_required() -> None:
    with pytest.raises(TypeError):
        SummarizationMiddleware(CountingChat())  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        SummarizationMiddleware(  # ty: ignore[missing-argument]
            CountingChat(), trigger=MessageCountTrigger(max_messages=1)
        )


async def test_summarize_context_tool_returns_result_without_state_mutation() -> None:
    middleware = SummarizationMiddleware(
        CountingChat(),
        provide_tool=True,
        trigger=MessageCountTrigger(max_messages=3),
        recent=RecentMessagesPolicy(keep_recent=1),
    )

    initial = _humans(4)
    state = {"messages": list(initial)}
    ctx = ToolContext(state=state)
    (summarize_context,) = middleware.tools()

    raw: Any = (await call_tool(summarize_context, {}, ctx)).value
    result: SummarizationResult = raw

    assert summarize_context.name == "summarize_context"
    assert result.summary == "SUMMARY"
    assert result.original_message_count == 4
    assert result.summarized_message_count == 3
    assert result.kept_message_count == 1
    assert state["messages"] == initial


async def test_summarize_context_is_bound_to_its_middleware_instance() -> None:
    def summarization(text: str) -> SummarizationMiddleware:
        return SummarizationMiddleware(
            CountingChat(text),
            provide_tool=True,
            trigger=MessageCountTrigger(max_messages=3),
            recent=RecentMessagesPolicy(keep_recent=1),
        )

    first, second = summarization("FIRST"), summarization("SECOND")
    ctx = ToolContext(state={"messages": _humans(4)}, middleware=(second, first))

    raw: Any = (await call_tool(first.tools()[0], {}, ctx)).value
    result: SummarizationResult = raw

    assert result.summary == "FIRST"
    assert first.tools() == first.tools()
    assert first.tools()[0] is not second.tools()[0]
    assert _summarizer(CountingChat(), max_messages=3, keep_recent=1).tools() == ()


class _FixedTokenCounter:
    def __init__(self, per_message: int) -> None:
        self.per_message = per_message

    def count_message(self, message: BaseMessage) -> int:
        return self.per_message

    def count_messages(self, messages: list[BaseMessage]) -> int:
        return len(messages) * self.per_message


def test_recent_tokens_policy_keeps_token_budget() -> None:
    counter = _FixedTokenCounter(per_message=10)
    policy = RecentTokensPolicy(keep_recent_tokens=25)
    messages: list[BaseMessage] = [HumanMessage(content=f"msg {i}") for i in range(5)]
    older, recent = policy.split(messages, counter)
    assert len(recent) == 2
    assert len(older) == 3
    assert recent == messages[-2:]


def test_recent_tokens_policy_keeps_one_when_message_exceeds_budget() -> None:
    counter = _FixedTokenCounter(per_message=100)
    policy = RecentTokensPolicy(keep_recent_tokens=10)
    messages: list[BaseMessage] = [HumanMessage(content=f"msg {i}") for i in range(3)]
    older, recent = policy.split(messages, counter)
    assert recent == messages[-1:]
    assert older == messages[:-1]


def test_token_limit_trigger_needs_the_models_input_limit() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="max_input_tokens"):
        TokenLimitTrigger()  # ty: ignore[missing-argument]

    trigger = TokenLimitTrigger(max_input_tokens=1000)
    counter = _FixedTokenCounter(per_message=801)
    decision = trigger.should_summarize([HumanMessage(content="x")], counter)
    assert decision.should_summarize is True
    assert decision.token_count == 801

    counter = _FixedTokenCounter(per_message=800)
    decision = trigger.should_summarize([HumanMessage(content="x")], counter)
    assert decision.should_summarize is False


def test_approximate_token_counter_counts_text_and_tool_calls() -> None:
    counter = ApproximateTokenCounter()
    plain = HumanMessage(content="abcd")
    assert counter.count_message(plain) >= 1

    ai_message = AIMessage(
        content="hello",
        tool_calls=[
            ToolCall(id="1", name="lookup", arguments={"q": "answer"}),
        ],
    )
    tool = ToolMessage(content="result", name="lookup", tool_call_id="1")
    total = counter.count_messages([plain, ai_message, tool])
    assert total > counter.count_message(plain)


def test_approximate_token_counter_returns_at_least_one_for_empty() -> None:
    counter = ApproximateTokenCounter()
    assert counter.count_message(HumanMessage(content="")) >= 1


def _pairs_are_intact(messages: list) -> bool:
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            owner = next(
                (
                    candidate
                    for candidate in reversed(messages[:index])
                    if not isinstance(candidate, ToolMessage)
                ),
                None,
            )
            if not isinstance(owner, AIMessage) or message.tool_call_id not in {
                tool_call.id for tool_call in owner.tool_calls
            }:
                return False
        if isinstance(message, AIMessage) and message.tool_calls:
            answered = set()
            for follower in messages[index + 1 :]:
                if not isinstance(follower, ToolMessage):
                    break
                answered.add(follower.tool_call_id)
            if answered and answered != {
                tool_call.id for tool_call in message.tool_calls
            }:
                return False
    return True


def test_conflict_error_names_replacement_writes() -> None:
    from nodestep import Replace
    from nodestep.exceptions import InvalidUpdateError
    from nodestep.state import StateSchema, merge_parallel_updates

    schema = StateSchema.from_type(Convo)

    with pytest.raises(InvalidUpdateError, match="replace"):
        merge_parallel_updates(
            schema,
            {"messages": []},
            [
                ("a", {"messages": Replace([])}),
                ("b", {"messages": Replace([])}),
            ],
        )
    with pytest.raises(InvalidUpdateError) as info:
        merge_parallel_updates(
            schema,
            {"messages": []},
            [
                ("a", {"messages": Replace([])}),
                ("b", {"messages": Replace([])}),
            ],
        )
    assert "no merging reducer" not in str(info.value)


def test_fixed_token_trigger() -> None:
    trigger = FixedTokenTrigger(max_tokens=50)
    counter = _FixedTokenCounter(per_message=20)

    assert trigger.should_summarize(_humans(3), counter).should_summarize
    assert not trigger.should_summarize(_humans(2), counter).should_summarize


async def test_the_split_never_separates_a_call_from_its_result() -> None:
    model = ScriptedChat(["ok"])
    graph = _one_model_graph(
        [_summarizer(CountingChat(), max_messages=3, keep_recent=1)], model
    )
    call = ToolCall(id="c", name="look")

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="one"),
                HumanMessage(content="two"),
                AIMessage(content=None, tool_calls=[call]),
                ToolMessage(content="r", name="look", tool_call_id="c"),
            ]
        }
    )

    sent = list(model.requests[0].messages)
    assert [type(message).__name__ for message in sent] == [
        "SystemMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert _pairs_are_intact(result.data["messages"])


async def test_a_paused_node_leaves_no_bookkeeping_behind() -> None:
    summarization = _summarizer(CountingChat(), max_messages=3, keep_recent=1)

    @node
    def ask(state: AgentState) -> dict[str, Any]:
        interrupt("approve?", id="approve")
        return {}

    graph = Graph(
        AgentState, middleware=[summarization], state_store=InMemoryStateStore()
    ).flow(START >> ask, ask >> END)

    for thread in ("a", "b", "c"):
        result = await graph.ainvoke({"messages": _humans(2)}, thread_id=thread)
        assert result.interrupts

    gc.collect()
    assert len(summarization._pending) == 0


async def test_a_finished_run_leaves_no_bookkeeping_behind() -> None:
    summarization = _summarizer(CountingChat(), max_messages=3, keep_recent=1)
    kept: list[Any] = []

    class KeepContext(Middleware):
        def before_node(self, ctx):
            kept.append(ctx)

    graph = _one_model_graph(
        [summarization, KeepContext()], ScriptedChat(["one", "two"])
    )

    await graph.ainvoke({"messages": _humans(2)})
    await graph.ainvoke({"messages": _humans(2)})

    assert len(kept) == 2
    assert len(summarization._pending) == 0


class _ConvoDict(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]
    final_text: str | None
    note: str


def _misspelled(chat: Any) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        chat,
        trigger=MessageCountTrigger(max_messages=2),
        recent=RecentMessagesPolicy(keep_recent=1),
        messages_field="mesages",
    )


@pytest.mark.parametrize("schema", [AgentState, _ConvoDict])
async def test_a_messages_field_the_state_does_not_declare_raises(schema: Any) -> None:
    summarizer = CountingChat()
    think = model_node("think", chat=ScriptedChat(default="reply"))
    graph = Graph(schema, middleware=[_misspelled(summarizer)]).flow(
        START >> think, think >> END
    )

    with pytest.raises(GraphConfigError, match="'mesages'"):
        await graph.ainvoke({"messages": _humans(6)})
    assert summarizer.requests == []


async def test_a_react_agent_with_a_misspelled_messages_field_raises() -> None:
    from nodestep import build_react_agent

    summarizer = CountingChat()
    agent = build_react_agent(
        ScriptedChat(default="reply"), tools=[], middleware=[_misspelled(summarizer)]
    )

    with pytest.raises(GraphConfigError, match="'mesages'"):
        await agent.ainvoke({"messages": _humans(6)})
    assert summarizer.requests == []


async def test_a_declared_messages_field_without_a_value_is_empty() -> None:
    @node
    def first(state: _ConvoDict) -> dict:
        return {"messages": _humans(4)}

    think = model_node("think", chat=ScriptedChat(["reply"]))
    graph = Graph(
        _ConvoDict,
        middleware=[_summarizer(CountingChat(), max_messages=2, keep_recent=1)],
    ).flow(START >> first, first >> think, think >> END)

    result = await graph.ainvoke({"note": "start"})

    assert _contents(result.data["messages"]) == ["SUMMARY", "m3", "reply"]


@pytest.mark.parametrize("state", [None, {"mesages": _humans(4)}, BaseState()])
async def test_summarize_state_without_the_messages_field_raises(state: Any) -> None:
    middleware = _summarizer(CountingChat(), max_messages=3, keep_recent=1)

    with pytest.raises(GraphConfigError, match="'messages'"):
        await middleware.summarize_state(state)

from typing import Annotated, Any, TypedDict

import pytest

from nodestep import END, START, Graph, node, tool, tool_runner
from nodestep.chat import Message, ToolCall
from nodestep.middleware import (
    InterruptRule,
    ToolInterruptMiddleware,
    all_fields,
    any_field,
    eq,
    ge,
    gt,
    le,
    lt,
    matches,
    should_interrupt,
)
from nodestep.utils.reducers import add_messages


def _asks(predicate: Any, arguments: dict[str, Any]) -> bool:
    return should_interrupt(
        [InterruptRule(tool="pay", match=predicate)], "pay", arguments
    )


@pytest.mark.parametrize(
    ("predicate", "amount", "expected"),
    [
        (gt("amount", 1000), 1500.0, True),
        (gt("amount", 1000), 1000, False),
        (ge("amount", 1000), 1000, True),
        (ge("amount", 1000), 999.5, False),
        (lt("amount", 0), -1, True),
        (lt("amount", 0), 0, False),
        (le("amount", 0), 0, True),
        (le("amount", 0), 0.5, False),
        (eq("amount", 5), 5, True),
        (eq("amount", 5), 5.0, True),
        (eq("amount", 5), 6, False),
    ],
)
def test_numeric_predicates_compare_numbers(
    predicate: Any, amount: float, expected: bool
) -> None:
    assert _asks(predicate, {"amount": amount}) is expected


def test_string_predicates_compare_strings() -> None:
    assert _asks(eq("to", "alice"), {"to": "alice"})
    assert not _asks(eq("to", "alice"), {"to": "bob"})
    assert _asks(matches("to", r"@evil\.com$"), {"to": "x@evil.com"})
    assert not _asks(matches("to", r"@evil\.com$"), {"to": "x@good.com"})


@pytest.mark.parametrize(
    "predicate",
    [
        gt("amount", 1000),
        ge("amount", 1000),
        lt("amount", 0),
        le("amount", 0),
        eq("amount", 5),
        matches("amount", r"^1"),
        all_fields(amount=r"^1"),
        any_field(amount=r"^1"),
    ],
)
def test_a_missing_argument_asks_the_human(predicate: Any) -> None:
    assert _asks(predicate, {"other": 1})


@pytest.mark.parametrize(
    ("predicate", "value"),
    [
        (matches("amount", r"^\d{4,}$"), 1500.0),
        (matches("flag", r"^$"), True),
        (gt("amount", 1000), "5000"),
        (gt("amount", 1000), True),
        (eq("amount", 1), True),
        (eq("to", "alice"), 1),
        (all_fields(amount=r"^0"), 1500),
        (any_field(amount=r"^0"), 1500),
    ],
)
def test_a_value_of_the_wrong_type_asks_the_human(predicate: Any, value: Any) -> None:
    assert _asks(predicate, {"amount": value, "to": value, "flag": value})


def test_matches_raises_type_error_for_non_strings() -> None:
    with pytest.raises(TypeError, match="amount"):
        matches("amount", r"^1")({"amount": 1000})


def test_numeric_predicates_raise_type_error_for_other_types() -> None:
    with pytest.raises(TypeError, match="amount"):
        gt("amount", 1000)({"amount": "5000"})
    with pytest.raises(TypeError, match="amount"):
        gt("amount", 1000)({"amount": True})


def test_missing_argument_raises_lookup_error() -> None:
    with pytest.raises(LookupError, match="amount"):
        gt("amount", 1000)({})


def test_a_predicate_that_raises_asks_the_human() -> None:
    def broken(arguments: Any) -> bool:
        raise RuntimeError("bug in the rule")

    assert _asks(broken, {"amount": 1})


def test_all_and_any_field_still_match_strings() -> None:
    both = all_fields(to=r"@external\.com$", subject=r"(?i)urgent")
    either = any_field(to=r"@evil\.com$", subject=r"(?i)spam")

    assert _asks(both, {"to": "u@external.com", "subject": "URGENT"})
    assert not _asks(both, {"to": "u@external.com", "subject": "hi"})
    assert _asks(either, {"to": "x@good.com", "subject": "SPAM"})
    assert not _asks(either, {"to": "x@good.com", "subject": "hi"})


class PayState(TypedDict, total=False):
    messages: Annotated[list[Message], add_messages]
    tool_calls: list[ToolCall]


paid: list[float] = []


@tool
def pay(amount: float, memo: str | None = None) -> str:
    """Pay."""
    paid.append(amount)
    return "paid"


def _pay_graph(rule: InterruptRule, arguments: dict[str, Any]) -> Graph:
    @node
    def request(state: PayState) -> dict:
        return {"tool_calls": [ToolCall(id="p1", name="pay", arguments=arguments)]}

    runner = tool_runner("tools", tools=[pay])
    return Graph(PayState, middleware=[ToolInterruptMiddleware(rules=[rule])]).flow(
        START >> request, request >> runner, runner >> END
    )


@pytest.mark.parametrize(
    ("rule", "arguments"),
    [
        (InterruptRule(tool="pay", match=gt("amount", 1000)), {"amount": 1500.0}),
        (InterruptRule(tool="pay", match=matches("memo", r"urgent")), {"amount": 1}),
        (InterruptRule(tool="pay", match=gt("amout", 1000)), {"amount": 1}),
    ],
)
async def test_the_approval_gate_fails_closed(
    rule: InterruptRule, arguments: dict[str, Any]
) -> None:
    paid.clear()

    result = await _pay_graph(rule, arguments).ainvoke({"messages": []})

    assert result.status == "interrupted"
    assert paid == []


async def test_a_rule_that_does_not_match_lets_the_call_run() -> None:
    paid.clear()
    rule = InterruptRule(tool="pay", match=gt("amount", 1000))

    result = await _pay_graph(rule, {"amount": 10}).ainvoke({"messages": []})

    assert result.status == "completed"
    assert paid == [10]


def test_all_fields_checks_every_argument_even_after_a_miss() -> None:
    rule = all_fields(to=r"@external\.com$", subjet=r"(?i)urgent")

    assert _asks(rule, {"to": "u@internal.com", "subject": "hi"})


@pytest.mark.parametrize(
    "predicate",
    [
        gt("amount", 100),
        ge("amount", 100),
        lt("amount", 0),
        le("amount", 0),
        eq("amount", 5),
    ],
)
def test_a_nan_argument_asks_the_human(predicate: Any) -> None:
    assert _asks(predicate, {"amount": float("nan")})


def test_numeric_predicates_raise_value_error_for_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        gt("amount", 100)({"amount": float("nan")})


def test_any_field_and_all_fields_need_a_pattern() -> None:
    with pytest.raises(ValueError, match="at least one"):
        any_field()
    with pytest.raises(ValueError, match="at least one"):
        all_fields()

from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

import pytest
from pydantic import Field

from nodestep import END, START, BaseState, Graph, add, add_messages, node
from nodestep.chat import HumanMessage, Message
from nodestep.state import StateSchema
from nodestep_sandbox.errors import FormError
from nodestep_sandbox.state_form import StateForm, type_label


class Ticket(BaseState):
    messages: Annotated[list[Message], add_messages] = Field(default_factory=list)
    title: str = "draft"
    count: int = 3
    ratio: float = 0.5
    ready: bool = True
    note: str | None = None
    tags: Annotated[list[str], add] = Field(default_factory=list)
    level: int


class Typed(TypedDict):
    text: str
    flag: bool


class Color(Enum):
    RED = "red"
    GREEN = "green"


class Choices(BaseState):
    choice: Literal["a", "b"] = "b"
    color: Color = Color.GREEN
    size: Literal[1, 2] | None = None


class TypedChoices(TypedDict):
    choice: Literal["a", "b"]
    color: Color


def form_for(schema_type: type) -> StateForm:
    return StateForm(StateSchema.from_type(schema_type))


def test_fields_describe_type_reducer_default_and_control() -> None:
    fields = {field.name: field for field in form_for(Ticket).fields}
    messages = fields["messages"]
    assert (messages.widget, messages.reducer, messages.default) == (
        "messages",
        "add_messages",
        "list()",
    )
    assert (
        messages.type == "list[SystemMessage | HumanMessage | AIMessage | ToolMessage]"
    )
    assert (
        fields["title"].widget,
        fields["title"].initial,
        fields["title"].default,
    ) == (
        "text",
        "draft",
        '"draft"',
    )
    assert (fields["count"].widget, fields["count"].initial) == ("integer", "3")
    assert (fields["ratio"].widget, fields["ratio"].initial) == ("number", "0.5")
    assert (fields["ready"].widget, fields["ready"].checked) == ("checkbox", True)
    assert (fields["note"].widget, fields["note"].type, fields["note"].default) == (
        "text",
        "str | None",
        "null",
    )
    assert (fields["tags"].widget, fields["tags"].reducer, fields["tags"].type) == (
        "json",
        "add",
        "list[str]",
    )
    assert fields["level"].default == "required"
    assert fields["level"].input_name == "state.level"


def test_typeddict_fields_have_no_defaults() -> None:
    fields = form_for(Typed).fields
    assert [(field.name, field.widget, field.default) for field in fields] == [
        ("text", "text", "no default"),
        ("flag", "checkbox", "no default"),
    ]


def test_untyped_state_has_no_fields() -> None:
    form = form_for(dict)
    assert form.untyped is True
    assert form.fields == []
    assert form.has_messages is False


def test_parse_converts_controls_and_leaves_empty_ones_out() -> None:
    value = form_for(Ticket).parse(
        {
            "state.messages": "Hello",
            "state.title": "",
            "state.count": "7",
            "state.ratio": "",
            "state.note": "hi",
            "state.tags": '["a"]',
            "state.level": "2",
        }
    )
    messages = value.pop("messages")
    assert [type(message) for message in messages] == [HumanMessage]
    assert messages[0].content == "Hello"
    assert value == {
        "count": 7,
        "ready": False,
        "note": "hi",
        "tags": ["a"],
        "level": 2,
    }


def test_a_checked_checkbox_is_true() -> None:
    assert form_for(Ticket).parse({"state.ready": "on"})["ready"] is True


def test_bad_numbers_and_json_are_reported_per_control() -> None:
    with pytest.raises(FormError) as error:
        form_for(Ticket).parse(
            {"state.count": "x", "state.ratio": "y", "state.tags": "[1,"}
        )
    assert error.value.errors == {
        "state.count": "Enter a whole number",
        "state.ratio": "Enter a number",
        "state.tags": "Not valid JSON: Expecting value (line 1, column 4)",
    }


def test_raw_json_must_be_an_object() -> None:
    assert StateForm.parse_raw('{"text": "hi"}') == {"text": "hi"}
    with pytest.raises(FormError) as not_object:
        StateForm.parse_raw("[]")
    assert not_object.value.errors == {
        "raw": "The input must be a JSON object of state fields"
    }
    with pytest.raises(FormError) as broken:
        StateForm.parse_raw("{")
    assert broken.value.errors["raw"].startswith("Not valid JSON:")


async def test_the_parsed_form_is_input_the_graph_accepts() -> None:
    @node
    def keep(state: Ticket) -> dict[str, object]:
        return {"tags": ["seen"]}

    graph = Graph(Ticket, name="ticket").flow(START >> keep, keep >> END)
    value = form_for(Ticket).parse(
        {"state.messages": "Hi", "state.count": "5", "state.level": "1"}
    )
    result = await graph.ainvoke(value)
    assert result.state.count == 5
    assert result.state.ready is False
    assert result.state.tags == ["seen"]
    assert result.state.messages[0].content == "Hi"


def test_literal_and_enum_fields_offer_their_values() -> None:
    fields = {field.name: field for field in form_for(Choices).fields}
    assert (
        fields["choice"].widget,
        fields["choice"].options,
        fields["choice"].initial,
    ) == ("select", ["'a'", "'b'"], "1")
    assert (
        fields["color"].widget,
        fields["color"].options,
        fields["color"].initial,
    ) == ("select", ["RED", "GREEN"], "1")
    assert (
        fields["size"].widget,
        fields["size"].options,
        fields["size"].initial,
    ) == ("select", ["1", "2"], "")


def test_a_choice_gives_the_value_itself() -> None:
    form = form_for(Choices)
    assert form.parse({"state.choice": "0", "state.color": "0", "state.size": ""}) == {
        "choice": "a",
        "color": Color.RED,
    }
    with pytest.raises(FormError) as error:
        form.parse({"state.choice": "2", "state.color": "red", "state.size": "-1"})
    assert error.value.errors == {
        "state.choice": "Choose one of the listed values",
        "state.color": "Choose one of the listed values",
        "state.size": "Choose one of the listed values",
    }


@pytest.mark.parametrize("schema", [Choices, TypedChoices])
async def test_chosen_values_are_input_the_graph_accepts(schema: type) -> None:
    @node
    def keep(state: Any) -> dict[str, object]:
        return {}

    graph = Graph(schema, name="choices").flow(START >> keep, keep >> END)
    value = form_for(schema).parse({"state.choice": "0", "state.color": "0"})
    result = await graph.ainvoke(value)
    state: Any = result.state
    if isinstance(state, dict):
        assert (state["choice"], state["color"]) == ("a", Color.RED)
    else:
        assert (state.choice, state.color) == ("a", Color.RED)


def test_type_label_reads_like_the_annotation() -> None:
    assert type_label(dict[str, list[int]]) == "dict[str, list[int]]"
    assert type_label(tuple[int, ...]) == "tuple[int, ...]"
    assert type_label(Annotated[int, "meta"] | None) == "int | None"

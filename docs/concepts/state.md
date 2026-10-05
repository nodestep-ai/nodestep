# State

The state is the typed data a graph works on: nodes read it and return updates, and reducers decide how each update merges in.

```python
from typing import Annotated

from pydantic import Field

from nodestep import END, START, BaseState, Graph, add, node


class Notes(BaseState):
    topic: str = ""
    lines: Annotated[list[str], add] = Field(default_factory=list)


@node
def write(state: Notes) -> dict:
    return {"topic": "release", "lines": ["tag v0.1.0"]}


graph = Graph(Notes).flow(START >> write, write >> END)

print(graph.invoke({"lines": ["update the changelog"]}).state)
```

```text
topic='release' lines=['update the changelog', 'tag v0.1.0']
```

The node returns a new `topic` and one line. `topic` has no reducer, so the update replaces it. `lines` uses `add`, so the line is appended to the input.

## State types

A state type describes the fields of the state. It is one of:

- `BaseState`, a pydantic `BaseModel` with the default configuration, or any other `BaseModel`
- a `TypedDict`
- `dict`, for untyped state

Nodes receive the state as the declared type. `result.state` is the declared type, and `result.data` is the same state as a dict.

```python
from typing import TypedDict

from nodestep import END, START, Graph, node


class Counter(TypedDict):
    count: int


@node
def increment(state: Counter) -> dict:
    return {"count": state["count"] + 1}


graph = Graph(Counter).flow(START >> increment, increment >> END)

print(graph.invoke({"count": 1}).state)
```

```text
{'count': 2}
```

The state holds only data that survives a JSON round trip. Clients and handles go in the [run context](nodes.md#run-context).

## Reducers

A reducer merges an update to a field with the field's current value. You set it with `Annotated`, as `lines` does above. nodestep has four built-in reducers:

- `replace`: the update replaces the value. Fields without a reducer use it.
- `add`: the items of a list update are appended.
- `add_messages`: chat messages are appended, and a message with a known id is edited.
- `merge_dict`: the top-level keys of a dict update are merged.

The exact rules are in [Rules and defaults](#rules-and-defaults), and the signatures in the [reducers reference](../reference/core/reducers.md).

## Custom reducers

A custom reducer is a function that takes the current value and the update, and returns the new value.

```python
from typing import Annotated, TypedDict

from nodestep import END, START, Graph, node


def join_lines(current: str | None, update: str) -> str:
    return update if current is None else f"{current}\n{update}"


class Log(TypedDict):
    text: Annotated[str, join_lines]


@node
def first(state: Log) -> dict:
    return {"text": "started"}


@node
def second(state: Log) -> dict:
    return {"text": "finished"}


graph = Graph(Log).flow(START >> first, first >> second, second >> END)

print(graph.invoke({}).state["text"])
```

```text
started
finished
```

The first write of a field that has no value yet passes `None` as the current value. Your reducer must handle it; nodestep's own reducers do.

## Bypass a reducer

`Replace(value)` sets a field without its reducer. `RemoveMessage(id)` in an `add_messages` update removes that message.

```python
from typing import Annotated, TypedDict

from nodestep import END, START, Graph, RemoveMessage, Replace, add_messages, node
from nodestep.chat import AIMessage, HumanMessage, Message


class Conversation(TypedDict):
    messages: Annotated[list[Message], add_messages]
    topics: list[str]


@node
def tidy(state: Conversation) -> dict:
    oldest = state["messages"][0]
    return {
        "messages": [RemoveMessage(oldest.id), AIMessage(content="Hello")],
        "topics": Replace(["greeting"]),
    }


graph = Graph(Conversation).flow(START >> tidy, tidy >> END)

result = graph.invoke(
    {"messages": [HumanMessage(content="hi"), HumanMessage(content="there")]}
)
print([message.content for message in result.state["messages"]])
print(result.state["topics"])
```

```text
['there', 'Hello']
['greeting']
```

Both work in node updates, `Command(update=...)`, `graph.update_state(...)` and run input. They do not work in `Send` payloads.

## Validation

nodestep validates input and updates before it stores them. An invalid value raises `StateUpdateError`, and a value that does not survive a JSON round trip raises `StateStoreError`.

```python
from typing import Any

from nodestep import (
    END,
    START,
    BaseState,
    Graph,
    StateStoreError,
    StateUpdateError,
    node,
)


class Order(BaseState):
    quantity: int = 0
    note: Any = None


@node
def noop(state: Order) -> dict:
    return {}


graph = Graph(Order).flow(START >> noop, noop >> END)

print(graph.invoke({"quantity": "7"}).state.quantity)
for bad_input in [{"qty": 1}, {"quantity": "seven"}, {"note": object()}]:
    try:
        graph.invoke(bad_input)
    except (StateUpdateError, StateStoreError) as error:
        print(type(error).__name__)
```

```text
7
StateUpdateError
StateUpdateError
StateStoreError
```

Validation uses pydantic's lax mode, so `"7"` becomes `7`. `qty` is not a field, `"seven"` is not a number, and an `object()` has no JSON form. Strict mode and the rules for stored values are in [Rules and defaults](#rules-and-defaults).

## Input of a run

The input of a run is merged into the state like an update. On a new thread, it is merged into the schema defaults. On an existing thread, it is a patch on the stored state; [Persistence and time travel](persistence.md) explains threads.

A dict sets only the fields it names. A model instance is the full state, so it replaces every field.

## Rules and defaults

| Rule or setting | What happens |
|---|---|
| `BaseState` | A pydantic `BaseModel` with pydantic's default configuration |
| `dict` as the state type | Accepts any key |
| `TypedDict` field that was never set | Has no value |
| What the state holds | Only data that survives a JSON round trip; clients and handles go in `context=` |
| Field without a reducer | `replace`, the default reducer: the update replaces the value |
| `add` | Appends the items of a list update; wrap a single item in a list |
| `add_messages`, message without an id | The message gets a `uuid4` hex id, on a copy; your object is not changed |
| `add_messages`, known id | A message with the same id replaces the existing one in place, so returning a message again edits it |
| `merge_dict` | Shallow: merges the top-level keys of a dict update. It cannot remove a key; use `Replace(value)` |
| First write of a field with no value | The reducer receives `None` on the first write, for example on an unset `TypedDict` field; nodestep's own reducers handle it |
| Two reducers on one field | `GraphConfigError` when the graph is created; so does a reducer inside `Optional[...]` instead of around it, or a callable without two positional parameters |
| `Replace` and `RemoveMessage` | Work in node updates, `Command(update=...)`, `graph.update_state(...)` and run input, not in `Send` payloads |
| Invalid value | `StateUpdateError`, naming the field and the writer |
| Value that does not survive a JSON round trip | `StateStoreError` |
| Validation mode | pydantic's lax mode: `"7"` becomes `7` for an `int` field |
| Strict mode | Set `model_config = ConfigDict(strict=True)` on the state model, or `@with_config(ConfigDict(strict=True))` on a `TypedDict` |
| What a field is validated with | Each field on its own, with its type, its `Annotated` metadata and the schema's pydantic config |
| How values are stored | Every value is stored as JSON through its field's type, with or without a state store, and must read back equal, with the same types |
| Aliases and computed fields | Stored models keep their aliases and leave out computed fields |
| Fields with `exclude=True` and private attributes | Not stored, so a model is refused while they hold anything but their defaults |
| `SecretStr` | Refused anywhere in a value; pass secrets through `context=` |
| A storage refusal | Names the place in the value and the types, never the values |
| Input on a new thread | Merged into the schema defaults |
| Input on an existing thread | A patch, merged into the stored state through the reducers |
| Dict input | Sets the fields it names |
| Model instance as input | The full state: it replaces every field, so passing `result.state` back changes nothing |
| Input keys that are not state fields | `StateUpdateError` |

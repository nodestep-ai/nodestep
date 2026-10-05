import json
from collections.abc import Mapping
from enum import Enum
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field

from nodestep import add_messages
from nodestep.chat import HumanMessage
from nodestep.state import FieldDescriptor, StateSchema, to_json_value
from nodestep_sandbox.errors import FormError
from nodestep_sandbox.records import parse_json

Widget = Literal["text", "integer", "number", "checkbox", "select", "messages", "json"]
_NO_VALUE = object()


def type_label(hint: Any) -> str:
    """Return a short readable name for a type annotation.

    ``Annotated`` metadata is left out and ``typing.`` prefixes are dropped.

    Parameters
    ----------
    hint : Any

    Returns
    -------
    str
    """
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Annotated:
        return type_label(args[0])
    if origin in (Union, UnionType):
        return " | ".join(type_label(argument) for argument in args)
    if origin is Literal:
        return f"Literal[{', '.join(repr(argument) for argument in args)}]"
    if hint is NoneType:
        return "None"
    if hint is Ellipsis:
        return "..."
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        if not args:
            return name
        return f"{name}[{', '.join(type_label(argument) for argument in args)}]"
    if isinstance(hint, type):
        return hint.__name__
    return str(hint).replace("typing.", "")


def _bare(hint: Any) -> Any:
    while get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return hint


def _optional_base(hint: Any) -> Any:
    hint = _bare(hint)
    if get_origin(hint) in (Union, UnionType):
        members = [argument for argument in get_args(hint) if argument is not NoneType]
        if len(members) == 1:
            return _bare(members[0])
    return hint


def _choices(hint: Any) -> list[Any]:
    base = _optional_base(hint)
    if get_origin(base) is Literal:
        return list(get_args(base))
    if isinstance(base, type) and issubclass(base, Enum):
        return list(base)
    return []


class StateField(BaseModel):
    """A state field as the pages describe it and the new-run form asks for it.

    Attributes
    ----------
    name : str
    type : str
        Readable form of the field's type.
    reducer : str
        Name of the reducer that merges updates into the field.
    default : str
        The default as JSON, the default factory as ``name()``, ``required``,
        or ``no default`` for TypedDict fields.
    widget : {"text", "integer", "number", "checkbox", "select", "messages", "json"}
        Form control; ``messages`` is a box for one human message, ``select``
        a list of the values a ``Literal`` or ``Enum`` field allows.
    initial : str
        Initial text of the control; for a select, the position of the
        default in ``choices``.
    checked : bool
        Initial state of a checkbox.
    choices : list[Any]
        The values a select offers.
    """

    name: str
    type: str
    reducer: str
    default: str
    widget: Widget
    initial: str = ""
    checked: bool = False
    choices: list[Any] = Field(default_factory=list)

    @property
    def input_name(self) -> str:
        """Name of the form control."""
        return f"state.{self.name}"

    @property
    def options(self) -> list[str]:
        """Labels of ``choices``: an enum member's name, or the value's repr."""
        return [
            value.name if isinstance(value, Enum) else repr(value)
            for value in self.choices
        ]


class StateForm:
    """The new-run form generated from a graph's state schema.

    Parameters
    ----------
    schema : StateSchema

    Attributes
    ----------
    untyped : bool
        Whether the state is untyped ``dict``, which has no fields to ask for.
    fields : list[StateField]
    """

    def __init__(self, schema: StateSchema[Any]) -> None:
        self.untyped = schema.is_dynamic()
        self.fields = [
            _state_field(schema, descriptor) for descriptor in schema.fields.values()
        ]

    @property
    def has_messages(self) -> bool:
        """Whether a field uses ``add_messages``."""
        return any(field.widget == "messages" for field in self.fields)

    def parse(self, values: Mapping[str, str]) -> dict[str, Any]:
        """Build run input from the submitted controls.

        Empty controls are left out, so the state defaults apply. A checkbox
        always gives ``True`` or ``False``. A message box gives a list with one
        ``HumanMessage``. A select gives the chosen value itself.

        Parameters
        ----------
        values : Mapping[str, str]
            Submitted form values by control name.

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        FormError
            If a number or JSON control holds text that does not parse, or a
            select sends a value it does not offer.
        """
        result: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for field in self.fields:
            raw = values.get(field.input_name, "")
            if field.widget == "checkbox":
                result[field.name] = field.input_name in values
            elif raw.strip():
                try:
                    result[field.name] = _convert(field, raw)
                except ValueError as error:
                    errors[field.input_name] = str(error)
        if errors:
            raise FormError(errors)
        return result

    @staticmethod
    def parse_raw(text: str) -> dict[str, Any]:
        """Parse the raw JSON input.

        Parameters
        ----------
        text : str

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        FormError
            If the text is not JSON or not a JSON object.
        """
        try:
            value = parse_json(text)
        except ValueError as error:
            raise FormError({"raw": str(error)}) from None
        if not isinstance(value, dict):
            raise FormError({"raw": "The input must be a JSON object of state fields"})
        return value


def _state_field(schema: StateSchema[Any], descriptor: FieldDescriptor) -> StateField:
    widget = _widget(descriptor)
    choices = _choices(descriptor.annotation) if widget == "select" else []
    default, value = _default(schema, descriptor.name)
    return StateField(
        name=descriptor.name,
        type=type_label(descriptor.annotation),
        reducer=getattr(
            descriptor.reducer, "__name__", type(descriptor.reducer).__name__
        ),
        default=default,
        widget=widget,
        initial=str(choices.index(value))
        if value in choices
        else _initial(widget, value),
        checked=value is True,
        choices=choices,
    )


def _widget(descriptor: FieldDescriptor) -> Widget:
    if descriptor.reducer is add_messages:
        return "messages"
    if _choices(descriptor.annotation):
        return "select"
    base = _optional_base(descriptor.annotation)
    if base is bool:
        return "checkbox"
    if base is int:
        return "integer"
    if base is float:
        return "number"
    if base is str:
        return "text"
    return "json"


def _default(schema: StateSchema[Any], name: str) -> tuple[str, Any]:
    if schema.kind != "pydantic":
        return "no default", _NO_VALUE
    info = schema.schema_type.model_fields[name]
    if info.default_factory is not None:
        return f"{getattr(info.default_factory, '__name__', 'factory')}()", _NO_VALUE
    if info.is_required():
        return "required", _NO_VALUE
    return json.dumps(to_json_value(info.default, fallback=repr)), info.default


def _initial(widget: Widget, value: Any) -> str:
    if widget == "text" and isinstance(value, str):
        return value
    if (
        widget in ("integer", "number")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return str(value)
    return ""


def _convert(field: StateField, raw: str) -> Any:
    widget = field.widget
    if widget == "select":
        if not raw.isdigit() or int(raw) >= len(field.choices):
            raise ValueError("Choose one of the listed values")
        return field.choices[int(raw)]
    if widget == "integer":
        try:
            return int(raw)
        except ValueError:
            raise ValueError("Enter a whole number") from None
    if widget == "number":
        try:
            return float(raw)
        except ValueError:
            raise ValueError("Enter a number") from None
    if widget == "messages":
        return [HumanMessage(content=raw)]
    if widget == "json":
        return parse_json(raw)
    return raw

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import json
import math
import reprlib
import sys
import types
import typing
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Generic,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    PydanticUserError,
    Secret,
    SecretBytes,
    SecretStr,
    TypeAdapter,
    ValidationError,
)

from nodestep.exceptions import (
    GraphConfigError,
    GraphExecutionError,
    InvalidUpdateError,
    ResumeError,
    StateStoreError,
    StateUpdateError,
)
from nodestep.models.base import NodestepModel
from nodestep.state.context import (
    REMOVE_MESSAGE_KEY,
    REPLACE_DELTA_KEY,
    isolate,
)
from nodestep.utils.reducers import (
    Reducer,
    RemoveMessage,
    Replace,
    add,
    add_messages,
    ensure_message_ids,
    merge_dict,
    message_id,
    replace,
)

StateT = TypeVar("StateT")
"""The state type a ``StateSchema`` validates and builds."""


def _is_typeddict(cls: Any) -> bool:
    if typing.is_typeddict(cls):
        return True
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and all(
            hasattr(cls, attribute)
            for attribute in ("__total__", "__required_keys__", "__optional_keys__")
        )
    )


def _is_pydantic_model(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _model_state_dict(model: BaseModel) -> dict[str, Any]:
    return {name: model.__dict__[name] for name in type(model).model_fields}


def _build_adapter(annotation: Any, config: ConfigDict | None) -> TypeAdapter[Any]:
    if config is not None:
        try:
            return TypeAdapter(annotation, config=config)
        except PydanticUserError as error:
            if error.code != "type-adapter-config-unused":
                raise
    return TypeAdapter(annotation)


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """A state field: its validation types and its reducer.

    Parameters
    ----------
    name : str
        Field name.
    annotation : Any
        The field's type with its constraints and validators, without the
        reducer; every value of the field is validated with it.
    reducer : Reducer, optional
        Merges an update into the field; ``replace`` by default.
    update_annotation : Any, optional
        Type an update is validated with before a reducer other than
        ``replace`` merges it: ``annotation`` without the field's own
        constraints and validators, which apply to the merged value. ``None``
        means ``annotation``.
    config : ConfigDict, optional
        Pydantic config of the state schema, applied to both types.
    """

    name: str
    annotation: Any
    reducer: Reducer = replace
    update_annotation: Any = None
    config: ConfigDict | None = None
    _adapter: TypeAdapter[Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _update_adapter: TypeAdapter[Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def adapter(self) -> TypeAdapter[Any]:
        """``TypeAdapter`` for ``annotation``, built on first use."""
        adapter = self._adapter
        if adapter is None:
            adapter = _build_adapter(self.annotation, self.config)
            object.__setattr__(self, "_adapter", adapter)
        return adapter

    @property
    def update_adapter(self) -> TypeAdapter[Any]:
        """``TypeAdapter`` for ``update_annotation``, built on first use."""
        adapter = self._update_adapter
        if adapter is None:
            annotation = (
                self.annotation
                if self.update_annotation is None
                else self.update_annotation
            )
            adapter = _build_adapter(annotation, self.config)
            object.__setattr__(self, "_update_adapter", adapter)
        return adapter


_QUALIFIERS = tuple(
    qualifier
    for qualifier in (
        typing.Required,
        typing.NotRequired,
        getattr(typing, "ReadOnly", None),
        typing_extensions.Required,
        typing_extensions.NotRequired,
        typing_extensions.ReadOnly,
    )
    if qualifier is not None
)


def _strip_qualifiers(hint: Any) -> Any:
    while get_origin(hint) in _QUALIFIERS:
        hint = get_args(hint)[0]
    return hint


def _callable_name(value: Any) -> str:
    return getattr(value, "__name__", repr(value))


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None)
    if module is None or name is None:
        return repr(value)
    return f"{module}.{name}"


def _is_union(hint: Any) -> bool:
    origin = get_origin(hint)
    return origin is typing.Union or origin is types.UnionType


def _check_union(owner: str, name: str, hint: Any) -> None:
    if not _is_union(hint):
        return
    for member in get_args(hint):
        if get_origin(member) is Annotated and any(
            callable(meta) for meta in member.__metadata__
        ):
            raise GraphConfigError(
                f"State field '{name}' of {owner} has its reducer inside "
                "Optional/Union, where it would be ignored; put Optional inside "
                "Annotated, e.g. Annotated[list[str] | None, add]"
            )


def _check_reducer(owner: str, name: str, reducer: Any) -> None:
    if isinstance(reducer, type):
        raise GraphConfigError(
            f"State field '{name}' of {owner} has the class {reducer.__name__} in "
            "its Annotated metadata; a reducer is a function taking (current, "
            "update), and classes are not reducers"
        )
    try:
        signature = inspect.signature(reducer)
    except (TypeError, ValueError) as error:
        raise GraphConfigError(
            f"Cannot read the signature of the reducer {reducer!r} of state field "
            f"'{name}' of {owner}; wrap it in a function taking (current, update)"
        ) from error
    parameters = signature.parameters.values()
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    unsupported = [
        parameter
        for parameter in parameters
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL
        or (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        )
    ]
    with_default = [
        parameter
        for parameter in positional
        if parameter.default is not inspect.Parameter.empty
    ]
    if len(positional) != 2 or unsupported or with_default:
        raise GraphConfigError(
            f"The callable {_callable_name(reducer)} in the Annotated metadata of "
            f"state field '{name}' of {owner} is read as the field's reducer, but a "
            "reducer takes exactly two positional parameters without defaults "
            f"(current, update); its signature is {signature}"
        )


def _members(hint: Any) -> tuple[Any, ...]:
    if get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return get_args(hint) if _is_union(hint) else (hint,)


def _is_message_class(item: Any) -> bool:
    from nodestep.chat.messages import BaseMessage

    return (
        isinstance(item, type)
        and issubclass(item, BaseMessage)
        and not inspect.isabstract(item)
    )


def _holds_messages(hint: Any) -> bool:
    lists = [member for member in _members(hint) if member is not type(None)]
    return bool(lists) and all(
        get_origin(member) is list
        and len(get_args(member)) == 1
        and all(_is_message_class(item) for item in _members(get_args(member)[0]))
        for member in lists
    )


def _descriptor(
    owner: str,
    name: str,
    hint: Any,
    metadata: Sequence[Any],
    extra: Sequence[Any] = (),
    config: ConfigDict | None = None,
) -> FieldDescriptor:
    _check_union(owner, name, hint)
    reducers = [meta for meta in metadata if callable(meta)]
    if len(reducers) > 1:
        raise GraphConfigError(
            f"State field '{name}' of {owner} declares more than one reducer "
            f"({', '.join(_callable_name(meta) for meta in reducers)}); declare one"
        )
    reducer = reducers[0] if reducers else replace
    if reducers:
        _check_reducer(owner, name, reducer)
    if reducer is add_messages and not _holds_messages(hint):
        raise GraphConfigError(
            f"State field '{name}' of {owner} uses add_messages, but its type "
            f"{inspect.formatannotation(hint)} does not read stored items back as "
            "chat messages; declare it as a list of chat messages, e.g. "
            "Annotated[list[Message], add_messages]"
        )
    update_annotation = Annotated[hint, *extra] if extra else hint
    own = [meta for meta in metadata if meta is not reducer]
    annotation = Annotated[update_annotation, *own] if own else update_annotation
    return FieldDescriptor(
        name=name,
        annotation=annotation,
        reducer=reducer,
        update_annotation=update_annotation,
        config=config,
    )


def _raw_annotations(schema: Any) -> dict[str, Any]:
    if sys.version_info >= (3, 14):
        import annotationlib

        return annotationlib.get_annotations(
            schema, format=annotationlib.Format.FORWARDREF
        )
    return dict(schema.__annotations__)


def _unresolved_typeddict_fields(schema: Any) -> list[str]:
    unresolved: list[str] = []
    for name, annotation in _raw_annotations(schema).items():
        try:
            get_type_hints(
                SimpleNamespace(__annotations__={name: annotation}),
                include_extras=True,
            )
        except Exception:
            unresolved.append(name)
    return unresolved


def _field_subject(names: list[str]) -> str:
    if not names:
        return "the state fields"
    return f"state field {', '.join(repr(name) for name in names)}"


def _typeddict_hints(schema: Any) -> dict[str, Any]:
    try:
        return get_type_hints(schema, include_extras=True)
    except Exception as error:
        subject = _field_subject(_unresolved_typeddict_fields(schema))
        raise GraphConfigError(
            f"Cannot resolve the type of {subject} of {schema.__name__}: "
            f"{type(error).__name__}: {error}; define the types a state schema uses at "
            "module level"
        ) from error


def _typeddict_fields(schema: Any) -> dict[str, FieldDescriptor]:
    config = getattr(schema, "__pydantic_config__", None)
    fields: dict[str, FieldDescriptor] = {}
    for name, hint in _typeddict_hints(schema).items():
        hint = _strip_qualifiers(hint)
        metadata: Sequence[Any] = ()
        if get_origin(hint) is Annotated:
            metadata = hint.__metadata__
            hint = _strip_qualifiers(get_args(hint)[0])
        fields[name] = _descriptor(schema.__name__, name, hint, metadata, config=config)
    return fields


def _unresolved(annotation: Any) -> bool:
    if isinstance(annotation, str | ForwardRef):
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return False
    arguments = get_args(annotation)
    if origin is Annotated:
        arguments = arguments[:1]
    return any(_unresolved(argument) for argument in arguments)


def _pydantic_fields(schema: type[BaseModel]) -> dict[str, FieldDescriptor]:
    owner = schema.__name__
    if schema.model_config.get("extra") == "allow":
        raise GraphConfigError(
            f"State schema {owner} sets extra='allow'; attributes that are not "
            "declared fields are not state and would be lost. Declare every field, "
            "or use dict for untyped state"
        )
    if not schema.__pydantic_complete__:
        subject = _field_subject(
            [
                name
                for name, info in schema.model_fields.items()
                if _unresolved(info.annotation)
            ]
        )
        raise GraphConfigError(
            f"Cannot resolve the type of {subject} of {owner}: the model is not "
            "fully defined; define the types it uses first, or call "
            f"{owner}.model_rebuild() once they exist"
        )
    decorators = schema.__pydantic_decorators__
    validators = [
        *decorators.field_validators,
        *decorators.model_validators,
        *decorators.validators,
        *decorators.root_validators,
    ]
    if validators:
        raise GraphConfigError(
            f"State schema {owner} uses validator decorators "
            f"({', '.join(validators)}); each state field is validated on its own "
            "when it is written, with its type and Annotated metadata only, so "
            "these would not run. Validate a field in its annotation, e.g. "
            "Annotated[str, AfterValidator(check)], and check rules that span "
            "several fields in a node"
        )
    serializers = [*decorators.field_serializers, *decorators.model_serializers]
    if serializers:
        raise GraphConfigError(
            f"State schema {owner} uses serializer decorators "
            f"({', '.join(serializers)}); each state field is stored on its own, "
            "with its type and Annotated metadata only, so these would not run. "
            "Serialize a field in its annotation, e.g. "
            "Annotated[datetime, PlainSerializer(to_text)]"
        )
    config = schema.model_config
    fields: dict[str, FieldDescriptor] = {}
    for name, info in schema.model_fields.items():
        if info.exclude or info.exclude_if is not None:
            marker = "exclude=True" if info.exclude else "exclude_if"
            raise GraphConfigError(
                f"State field '{name}' of {owner} sets {marker}; every state "
                "field is stored, so pass a value that must not be stored through "
                "context= instead"
            )
        if info.default_factory_takes_validated_data:
            raise GraphConfigError(
                f"State field '{name}' of {owner} has a default_factory that reads "
                "the other fields; state defaults are set before the input is "
                "merged, so give the factory no arguments"
            )
        discriminator = info.discriminator
        if isinstance(discriminator, str):
            discriminator = Discriminator(discriminator)
        extra = () if discriminator is None else (discriminator,)
        fields[name] = _descriptor(
            owner, name, info.annotation, info.metadata, extra, config
        )
    return fields


def _is_stored_replace(value: Any) -> bool:
    return isinstance(value, dict) and len(value) == 1 and REPLACE_DELTA_KEY in value


def _is_stored_removal(value: Any) -> bool:
    return isinstance(value, dict) and len(value) == 1 and REMOVE_MESSAGE_KEY in value


def _has_removal(value: Any) -> bool:
    if isinstance(value, list):
        return any(isinstance(item, RemoveMessage) for item in value)
    return isinstance(value, RemoveMessage)


def _reducer_of(descriptor: FieldDescriptor | None) -> Reducer:
    return descriptor.reducer if descriptor is not None else replace


def _check_removal(key: str, value: Any, reducer: Reducer) -> None:
    if isinstance(value, Replace):
        if _has_removal(value.value):
            raise StateUpdateError(
                f"Field '{key}' got RemoveMessage inside Replace; Replace sets the "
                "field to its value as given, so leave the message out of that value"
            )
    elif _has_removal(value) and reducer is not add_messages:
        raise StateUpdateError(
            f"Field '{key}' got RemoveMessage, which works only in a list update to "
            "a field with the add_messages reducer"
        )


_MERGING_REDUCERS = (add, add_messages, merge_dict)


def _describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            if item["loc"]
            else item["msg"]
            for item in error.errors(include_url=False)
        )
    return f"{type(error).__name__}: {error}"


def _source(writer: str | None) -> str:
    return "" if writer is None else f" from {writer}"


def _keep(item: RemoveMessage) -> RemoveMessage:
    return item


def _around_removals(
    value: Any,
    convert: Callable[[Any], Any],
    removal: Callable[[RemoveMessage], Any],
) -> Any:
    if not (isinstance(value, list) and _has_removal(value)):
        return convert(value)
    converted = iter(
        convert([item for item in value if not isinstance(item, RemoveMessage)])
    )
    return [
        removal(item) if isinstance(item, RemoveMessage) else next(converted)
        for item in value
    ]


def _without_removals(value: Any) -> Any:
    if isinstance(value, list):
        return [item for item in value if not isinstance(item, RemoveMessage)]
    return value


def _update_value(descriptor: FieldDescriptor | None, value: Any) -> Any:
    if isinstance(value, Replace):
        inner = (
            value.value
            if descriptor is None
            else descriptor.adapter.validate_python(value.value)
        )
        return Replace(ensure_message_ids(inner))
    if descriptor is None:
        return ensure_message_ids(value)
    if descriptor.reducer is replace:
        return ensure_message_ids(descriptor.adapter.validate_python(value))
    return ensure_message_ids(
        _around_removals(value, descriptor.update_adapter.validate_python, _keep)
    )


def _trusted_value(descriptor: FieldDescriptor | None, value: Any) -> Any:
    if isinstance(value, Replace):
        return Replace(ensure_message_ids(value.value))
    return ensure_message_ids(value)


def _validates_merged(descriptor: FieldDescriptor) -> bool:
    return not (
        descriptor.reducer in _MERGING_REDUCERS
        and descriptor.annotation is descriptor.update_annotation
    )


def _reduce(
    key: str, writer: str | None, descriptor: FieldDescriptor, current: Any, value: Any
) -> Any:
    reducer = descriptor.reducer
    try:
        merged = reducer(current, value)
    except Exception as error:
        if current is None and reducer not in (replace, *_MERGING_REDUCERS):
            raise _invalid(
                key,
                writer,
                error,
                "; the field had no value yet, so its reducer "
                f"{_qualified_name(reducer)} got None as the current value; make "
                "the reducer handle None, or use one that does, such as nodestep.add",
            ) from error
        if not isinstance(error, _UPDATE_ERRORS):
            raise
        raise _invalid(key, writer, error) from error
    if not _validates_merged(descriptor):
        return merged
    try:
        return descriptor.adapter.validate_python(merged)
    except _UPDATE_ERRORS as error:
        raise _invalid(key, writer, error) from error


_UNTYPED: TypeAdapter[Any] = TypeAdapter(Any)
_SECRETS = (SecretStr, SecretBytes, Secret)


def _json_bytes(adapter: TypeAdapter[Any], value: Any) -> bytes:
    return adapter.dump_json(value, by_alias=True, round_trip=True, warnings=False)


def _parts(value: Any) -> list[Any]:
    if isinstance(value, BaseModel):
        return [
            *_model_state_dict(value).values(),
            *(value.__pydantic_extra__ or {}).values(),
            *(value.__pydantic_private__ or {}).values(),
        ]
    if isinstance(value, Mapping):
        return [*value.keys(), *value.values()]
    if isinstance(value, list | tuple | set | frozenset | deque):
        return list(value)
    if dataclasses.is_dataclass(type(value)):
        return [getattr(value, item.name, None) for item in dataclasses.fields(value)]
    return []


def _secret_in(value: Any) -> type | None:
    if isinstance(value, _SECRETS):
        return type(value)
    for item in _parts(value):
        found = _secret_in(item)
        if found is not None:
            return found
    return None


_EXACT = frozenset({str, bytes, int, bool, type(None)})


@dataclass(frozen=True, slots=True)
class _Difference:
    path: str
    reason: str

    def at(self, key: str) -> str:
        return f"at {key}{self.path}, {self.reason}" if self.path else self.reason


def _changed(path: str, kind: type) -> _Difference:
    return _Difference(path, f"a {kind.__name__} comes back with another value")


def _same_float(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)


def _leaf_form(value: Any) -> bytes | None:
    try:
        return _json_bytes(_UNTYPED, value)
    except (TypeError, ValueError):
        return None


def _same_leaf(left: Any, right: Any) -> bool:
    try:
        if left != right:
            return False
    except (TypeError, ValueError):
        return False
    return _leaf_form(left) == _leaf_form(right)


def _key_label(key: Any) -> str:
    if type(key) in (str, int):
        return reprlib.repr(key)
    return f"<{type(key).__name__}>"


def _model_difference(
    left: BaseModel, right: BaseModel, path: str
) -> _Difference | None:
    owner = type(left).__name__
    for name, info in type(left).model_fields.items():
        found = _difference(left.__dict__[name], right.__dict__[name], f"{path}.{name}")
        if found is None:
            continue
        if info.exclude or info.exclude_if is not None:
            marker = "exclude=True" if info.exclude else "exclude_if"
            return _Difference(
                f"{path}.{name}",
                f"a field with {marker} is not stored, so {owner} comes back "
                "without its value; pass values that must not be stored through "
                "context=",
            )
        return found
    left_extra = left.__pydantic_extra__ or {}
    right_extra = right.__pydantic_extra__ or {}
    if list(left_extra) != list(right_extra):
        return _Difference(path, f"{owner} comes back with other extra attributes")
    for name, item in left_extra.items():
        found = _difference(item, right_extra[name], f"{path}.{name}")
        if found is not None:
            return found
    left_private = left.__pydantic_private__ or {}
    right_private = right.__pydantic_private__ or {}
    for name in dict.fromkeys([*left_private, *right_private]):
        if (
            name in left_private
            and name in right_private
            and _difference(left_private[name], right_private[name]) is None
        ):
            continue
        return _Difference(
            f"{path}.{name}",
            f"a private attribute is not stored, so {owner} comes back without "
            "its value",
        )
    return None


def _dict_difference(
    left: dict[Any, Any], right: dict[Any, Any], path: str
) -> _Difference | None:
    kind = type(left).__name__
    if len(left) != len(right):
        return _Difference(
            path, f"a {kind} of {len(left)} entries comes back with {len(right)}"
        )
    for (left_key, left_item), (right_key, right_item) in zip(
        left.items(), right.items(), strict=True
    ):
        if _difference(left_key, right_key) is not None:
            left_kind, right_kind = type(left_key).__name__, type(right_key).__name__
            if left_kind != right_kind:
                return _Difference(
                    path, f"a {left_kind} key comes back as a {right_kind} key"
                )
            return _Difference(path, f"a {left_kind} key comes back as another key")
        found = _difference(left_item, right_item, f"{path}[{_key_label(left_key)}]")
        if found is not None:
            return found
    return None


def _sequence_difference(
    left: Sequence[Any] | deque[Any], right: Sequence[Any] | deque[Any], path: str
) -> _Difference | None:
    kind = type(left).__name__
    if len(left) != len(right):
        return _Difference(
            path, f"a {kind} of {len(left)} items comes back with {len(right)}"
        )
    if (
        isinstance(left, deque)
        and isinstance(right, deque)
        and left.maxlen != right.maxlen
    ):
        return _Difference(
            path,
            f"a deque with maxlen {left.maxlen} comes back with maxlen {right.maxlen}",
        )
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        found = _difference(left_item, right_item, f"{path}[{index}]")
        if found is not None:
            return found
    return None


def _take_identical(candidates: list[Any], item: Any) -> bool:
    for index, candidate in enumerate(candidates):
        if _difference(item, candidate) is None:
            del candidates[index]
            return True
    return False


def _set_difference(
    left: set[Any] | frozenset[Any], right: set[Any] | frozenset[Any], path: str
) -> _Difference | None:
    kind = type(left).__name__
    if len(left) != len(right):
        return _Difference(
            path, f"a {kind} of {len(left)} items comes back with {len(right)}"
        )
    unmatched: dict[int, list[Any]] = {}
    for item in right:
        unmatched.setdefault(hash(item), []).append(item)
    for item in left:
        candidates = unmatched.get(hash(item), [])
        if _take_identical(candidates, item):
            continue
        if candidates:
            return _difference(item, candidates[0], f"{path}{{...}}")
        return _Difference(
            f"{path}{{...}}", f"an item of the {kind} comes back as another value"
        )
    return None


def _dataclass_difference(left: Any, right: Any, path: str) -> _Difference | None:
    for item in dataclasses.fields(left):
        found = _difference(
            getattr(left, item.name, None),
            getattr(right, item.name, None),
            f"{path}.{item.name}",
        )
        if found is not None:
            return found
    return None


def _difference(left: Any, right: Any, path: str = "") -> _Difference | None:
    if left is right:
        return None
    kind = type(left)
    if kind is not type(right):
        return _Difference(
            path, f"a {kind.__name__} comes back as {type(right).__name__}"
        )
    if kind in _EXACT:
        return None if left == right else _changed(path, kind)
    if kind is float:
        return None if _same_float(left, right) else _changed(path, kind)
    if isinstance(left, BaseModel):
        return _model_difference(left, right, path)
    if isinstance(left, dict):
        return _dict_difference(left, right, path)
    if isinstance(left, list | tuple | deque):
        return _sequence_difference(left, right, path)
    if isinstance(left, set | frozenset):
        return _set_difference(left, right, path)
    if dataclasses.is_dataclass(kind):
        return _dataclass_difference(left, right, path)
    return None if _same_leaf(left, right) else _changed(path, kind)


def _unchanged(old: Any, new: Any) -> bool:
    return _difference(old, new) is None


_UNTYPED_VALUES = (
    "untyped state holds only str, int, float, bool, None, and lists and dicts "
    "with str keys; declare the field's type in a state schema to store other values"
)


def _check_storable(
    key: str, writer: str | None, adapter: TypeAdapter[Any], value: Any
) -> None:
    secret = _secret_in(value)
    if secret is not None:
        raise StateUpdateError(
            f"State field '{key}' got a secret ({secret.__name__}){_source(writer)}; "
            "state values are stored, so pass secrets through context= instead"
        )
    try:
        restored = adapter.validate_json(_json_bytes(adapter, value))
    except (TypeError, ValueError) as error:
        detail = f"it cannot be stored and read back ({_describe(error)})"
    else:
        found = _difference(value, restored)
        if found is None:
            return
        detail = found.at(key)
    if adapter is _UNTYPED:
        raise StateStoreError(
            f"State field '{key}' got a value{_source(writer)} that does not survive "
            f"being stored as JSON in untyped state: {detail}; {_UNTYPED_VALUES}"
        )
    raise StateStoreError(
        f"State field '{key}' got a value{_source(writer)} that does not survive "
        f"being stored as JSON through its type: {detail}"
    )


def _dump(adapter: TypeAdapter[Any], value: Any) -> Any:
    return json.loads(_json_bytes(adapter, value))


def _read(adapter: TypeAdapter[Any], stored: Any) -> Any:
    return adapter.validate_json(json.dumps(stored))


def _read_reduced(descriptor: FieldDescriptor, stored: Any) -> Any:
    adapter = descriptor.update_adapter
    if descriptor.reducer is not add_messages or not isinstance(stored, list):
        return _read(adapter, stored)
    messages = iter(
        _read(adapter, [item for item in stored if not _is_stored_removal(item)])
    )
    return [
        RemoveMessage(item[REMOVE_MESSAGE_KEY])
        if _is_stored_removal(item)
        else next(messages)
        for item in stored
    ]


def _stored_removal(item: RemoveMessage) -> dict[str, str]:
    return {REMOVE_MESSAGE_KEY: item.id}


_UPDATE_ERRORS = (TypeError, ValueError, StateUpdateError)


def _invalid(
    key: str, writer: str | None, error: Exception, hint: str = ""
) -> StateUpdateError:
    return StateUpdateError(
        f"State field '{key}' got an invalid value{_source(writer)}: "
        f"{_describe(error)}{hint}"
    )


@dataclass(frozen=True)
class StateSchema(Generic[StateT]):
    """Fields and reducers of a state type.

    Validators and reducers may run more than once per write, so keep them
    free of side effects. See [Validation](../../concepts/state.md#validation)
    for how values are validated and stored.

    Attributes
    ----------
    schema_type : type
        The state type: a TypedDict, a pydantic model or ``dict``.
    fields : dict[str, FieldDescriptor]
        The fields by name; empty for ``dict``.
    kind : str
        ``"typeddict"``, ``"pydantic"`` or ``"dict"``.
    """

    schema_type: Any
    fields: dict[str, FieldDescriptor]
    kind: str

    @classmethod
    def from_type(cls, schema: Any) -> StateSchema[Any]:
        """Build a schema from a TypedDict, a pydantic model or ``dict``.

        ``dict`` means untyped state: any key, no reducers, JSON values only.
        A field's reducer is the one callable in its ``Annotated`` metadata,
        as in ``Annotated[list[str], add]``; the rest of the metadata is kept
        for validation. Class variables and private attributes of a pydantic
        schema are not fields.

        Parameters
        ----------
        schema : type
            A TypedDict, a pydantic ``BaseModel`` subclass, or ``dict``.

        Returns
        -------
        StateSchema

        Raises
        ------
        GraphConfigError
            If ``schema`` is anything else (``None``, ``Any`` and dict
            subclasses included), declares no fields or has a type that cannot
            be resolved; if a field has more than one reducer, a reducer that
            does not take exactly two positional parameters without defaults
            (``current, update``), a reducer inside ``Optional`` or ``Union``,
            or ``add_messages`` on a type that is not a list of chat messages;
            or if a pydantic schema sets
            ``extra='allow'``, uses validator or serializer decorators, has a
            ``default_factory`` that takes the validated data, or has a field
            with ``exclude=True`` or ``exclude_if``.
        """
        if schema is dict:
            return cls(schema_type=dict, fields={}, kind="dict")
        if _is_typeddict(schema):
            kind, fields = "typeddict", _typeddict_fields(schema)
        elif _is_pydantic_model(schema):
            kind, fields = "pydantic", _pydantic_fields(schema)
        else:
            raise GraphConfigError(
                f"Unsupported state schema {schema!r}; use a TypedDict, a pydantic "
                "BaseModel subclass, or dict for untyped state"
            )
        if not fields:
            raise GraphConfigError(
                f"State schema {schema.__name__} declares no fields; declare its "
                "fields, or use dict for untyped state"
            )
        return cls(schema_type=schema, fields=fields, kind=kind)

    def is_dynamic(self) -> bool:
        """Whether the schema is untyped ``dict`` state, which accepts any key."""
        return self.kind == "dict"

    @property
    def _name(self) -> str:
        return getattr(self.schema_type, "__name__", repr(self.schema_type))

    def _adapter(self, key: str) -> TypeAdapter[Any]:
        descriptor = self.fields.get(key)
        return _UNTYPED if descriptor is None else descriptor.adapter

    def to_declared(self, state: dict[str, Any]) -> Any:
        """Convert a state dict to the declared type, without validating it again."""
        if self.kind == "pydantic":
            return self.schema_type.model_construct(**state)
        return dict(state)

    def to_dict(self, value: Any) -> dict[str, Any]:
        """Convert a state value to a dict.

        Parameters
        ----------
        value : dict, BaseModel or None

        Returns
        -------
        dict
        """
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            return _model_state_dict(value)
        if isinstance(value, dict):
            return dict(value)
        raise StateUpdateError(
            f"State must be a dict or pydantic instance; got {type(value).__name__}"
        )

    def defaults(self) -> dict[str, Any]:
        """Default values of the fields that have one."""
        return self._defaults({})

    def _defaults(self, present: Mapping[str, Any]) -> dict[str, Any]:
        if self.kind != "pydantic":
            return {}
        return {
            name: info.get_default(call_default_factory=True)
            for name, info in self.schema_type.model_fields.items()
            if not info.is_required() and name not in present
        }

    def _replaces(self, update: Mapping[str, Any], key: str) -> bool:
        return key in update and (
            isinstance(update[key], Replace)
            or _reducer_of(self.fields.get(key)) is replace
        )

    def merge_input(
        self, current: dict[str, Any] | None, value: Any, *, writer: str = "the input"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge the input of a run into a thread's state through the reducers.

        The input is validated and merged like a node's update, on new and
        existing threads alike; a new thread starts from the schema defaults.
        An instance of the pydantic schema is the full state: it replaces
        every field, bypassing the reducers, and its values are validated
        again only if the schema sets ``revalidate_instances``. Partial input
        is a dict.

        Parameters
        ----------
        current : dict or None
            The thread's state, or None for a new thread.
        value : dict, BaseModel or None
            The input; None writes nothing.
        writer : str, optional
            Who wrote the input, as named in error messages.

        Returns
        -------
        tuple[dict, dict]
            The new state, and the validated update that was merged.

        Raises
        ------
        StateUpdateError
            If the input is neither a dict nor an instance of the pydantic
            schema, has an unknown key, an invalid value, a value its reducer
            rejects or a secret, or leaves a required field of a new thread
            without a value.
        StateStoreError
            If an input value, or a default a new thread keeps, does not
            survive being stored as JSON through its field's type.
        """
        if self.kind == "pydantic" and isinstance(value, self.schema_type):
            given = {
                key: item
                if _reducer_of(self.fields.get(key)) is replace
                else Replace(item)
                for key, item in _model_state_dict(
                    self.schema_type.model_validate(value)
                ).items()
            }
            read = _trusted_value
        else:
            given = self._input_update(value)
            read = _update_value
        base = current
        if base is None:
            base = self.defaults()
            for key, default in base.items():
                if not self._replaces(given, key):
                    _check_storable(key, "its default", self._adapter(key), default)
        update = self._checked_update(base, given, writer, read) or {}
        state, _ = merge_updates(self, base, update)
        if current is None and self.kind == "pydantic":
            missing = [
                name
                for name, info in self.schema_type.model_fields.items()
                if info.is_required() and name not in state
            ]
            if missing:
                raise StateUpdateError(
                    f"A new thread gets no value{_source(writer)} for the required "
                    f"state field {', '.join(repr(name) for name in missing)} of "
                    f"{self._name}"
                )
        return state, update

    def _input_update(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        accepted = (
            f"a dict or a {self._name} instance"
            if self.kind == "pydantic"
            else "a dict"
        )
        raise StateUpdateError(
            f"The input must be {accepted}, got {type(value).__name__}; a partial "
            "input is a dict of the fields it sets"
        )

    def validate_update(
        self,
        current: dict[str, Any],
        update: dict[str, Any] | None,
        *,
        writer: str,
    ) -> dict[str, Any] | None:
        """Validate an update against the state it will be merged into.

        A value of a ``replace`` field, or in ``Replace``, is validated as the
        field's value. A value for another reducer is validated with
        ``update_annotation`` and merged into ``current`` to check that it
        applies; the merged value is validated too when the field has its own
        constraints or validators, or a custom reducer. Chat messages without
        an id get one, and the result holds copies of the values.

        Parameters
        ----------
        current : dict
            The state the update will be merged into.
        update : dict or None
        writer : str
            Who wrote the update, for error messages, e.g. ``"node 'plan'"``.

        Returns
        -------
        dict or None
            The validated, copied update, ready to merge and store.

        Raises
        ------
        StateUpdateError
            If a key is not a state field; a value is invalid, rejected by its
            reducer, holds a secret or cannot be copied; a value is a dict
            whose only key is ``"__nodestep_replace__"``; or a
            ``RemoveMessage`` is inside ``Replace`` or sent to a field whose
            reducer is not ``add_messages``.
        StateStoreError
            If a value does not survive being stored as JSON through its type.
        """
        return self._checked_update(current, update, writer, _update_value)

    def _checked_update(
        self,
        current: dict[str, Any],
        update: dict[str, Any] | None,
        writer: str,
        read: Callable[[FieldDescriptor | None, Any], Any],
    ) -> dict[str, Any] | None:
        if not update:
            return update
        validated: dict[str, Any] = {}
        for key, value in update.items():
            descriptor = self.fields.get(key)
            if descriptor is None and not self.is_dynamic():
                raise StateUpdateError(
                    f"Unknown field '{key}' in state update{_source(writer)}"
                )
            if _is_stored_replace(value):
                raise StateUpdateError(
                    f"Field '{key}' got a dict with the reserved key "
                    f"'{REPLACE_DELTA_KEY}'; use Replace(value) to bypass the "
                    "field's reducer"
                )
            _check_removal(key, value, _reducer_of(descriptor))
            try:
                checked = read(descriptor, value)
            except _UPDATE_ERRORS as error:
                raise _invalid(key, writer, error) from error
            if isinstance(checked, Replace):
                _check_storable(key, writer, self._adapter(key), checked.value)
            elif descriptor is None or descriptor.reducer is replace:
                _check_storable(key, writer, self._adapter(key), checked)
            else:
                _check_storable(
                    key, writer, descriptor.update_adapter, _without_removals(checked)
                )
                merged = _reduce(key, writer, descriptor, current.get(key), checked)
                if _validates_merged(descriptor):
                    _check_storable(key, writer, descriptor.adapter, merged)
            validated[key] = checked
        return isolate(validated)

    def validate_payload(
        self, payload: dict[str, Any], *, writer: str
    ) -> dict[str, Any]:
        """Validate the values of a ``Send`` payload as field values.

        A payload bypasses the reducers, so each value is validated as its
        field's value. The result holds copies of the values.

        Parameters
        ----------
        payload : dict
            Values keyed by state field.
        writer : str
            Who sent the payload, for error messages.

        Returns
        -------
        dict

        Raises
        ------
        StateUpdateError
            If a value is not valid for its field, holds a secret or cannot be
            copied.
        StateStoreError
            If a value does not survive being stored as JSON through its type.
        """
        validated: dict[str, Any] = {}
        for key, value in payload.items():
            descriptor = self.fields.get(key)
            try:
                checked = (
                    value
                    if descriptor is None
                    else descriptor.adapter.validate_python(value)
                )
            except ValidationError as error:
                raise StateUpdateError(
                    f"State field '{key}' got an invalid value{_source(writer)}: "
                    f"{_describe(error)}"
                ) from error
            _check_storable(key, writer, self._adapter(key), checked)
            validated[key] = checked
        return isolate(validated)

    def dump_state(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a state, or part of one such as a ``Send`` payload, to JSON data.

        ``restore_state`` and ``restore_payload`` read the result back.

        Parameters
        ----------
        values : Mapping
            Validated values keyed by state field.

        Returns
        -------
        dict
        """
        return {key: _dump(self._adapter(key), value) for key, value in values.items()}

    def dump_update(self, update: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Convert a validated update to JSON data, as it is stored in history.

        ``Replace(value)`` becomes ``{"__nodestep_replace__": value}`` and
        ``RemoveMessage(id)`` becomes ``{"__nodestep_remove__": id}``.
        ``restore_update`` reads the result back.

        Parameters
        ----------
        update : Mapping or None
            An update from ``validate_update``.

        Returns
        -------
        dict or None
        """
        if update is None:
            return None
        dumped: dict[str, Any] = {}
        for key, value in update.items():
            descriptor = self.fields.get(key)
            if isinstance(value, Replace):
                dumped[key] = {
                    REPLACE_DELTA_KEY: _dump(self._adapter(key), value.value)
                }
            elif descriptor is None or descriptor.reducer is replace:
                dumped[key] = _dump(self._adapter(key), value)
            else:
                adapter = descriptor.update_adapter
                dumped[key] = _around_removals(
                    value, functools.partial(_dump, adapter), _stored_removal
                )
        return dumped

    def _check_stored_keys(self, stored: Mapping[str, Any]) -> None:
        if self.is_dynamic():
            return
        unknown = [key for key in stored if key not in self.fields]
        if unknown:
            raise StateStoreError(
                "The store holds values for "
                f"{', '.join(repr(key) for key in unknown)}, which state schema "
                f"{self._name} does not declare; the schema changed after the "
                "thread was stored"
            )

    def _restore_values(self, stored: Mapping[str, Any]) -> dict[str, Any]:
        self._check_stored_keys(stored)
        restored: dict[str, Any] = {}
        for key, value in stored.items():
            descriptor = self.fields.get(key)
            try:
                restored[key] = (
                    value if descriptor is None else _read(descriptor.adapter, value)
                )
            except ValidationError as error:
                raise ResumeError(
                    f"The stored value of state field '{key}' is not valid for state "
                    f"schema {self._name}: {_describe(error)}"
                ) from error
        return restored

    def restore_state(self, stored: Mapping[str, Any]) -> dict[str, Any]:
        """Read a stored state, such as a checkpoint, back into the field types.

        Fields missing from the stored state, such as fields added to the
        schema later, get their defaults; no other default factory is called.

        Parameters
        ----------
        stored : Mapping
            The state as it was stored.

        Returns
        -------
        dict

        Raises
        ------
        StateStoreError
            If the stored state has a key that is not a state field.
        ResumeError
            If a stored value is not valid for its field.
        """
        restored = self._restore_values(stored)
        return {**self._defaults(restored), **restored}

    def restore_payload(self, stored: Mapping[str, Any]) -> dict[str, Any]:
        """Read a stored ``Send`` payload back into the field types.

        Parameters
        ----------
        stored : Mapping
            The payload as it was stored.

        Returns
        -------
        dict

        Raises
        ------
        StateStoreError
            If the payload has a key that is not a state field.
        ResumeError
            If a stored value is not valid for its field.
        """
        return self._restore_values(stored)

    def restore_update(self, update: Mapping[str, Any]) -> dict[str, Any]:
        """Read an update from a stored event back as a validated update.

        Reverses ``dump_update``, turning the stored forms back into
        ``Replace`` and ``RemoveMessage``. The result is ready for
        ``merge_updates``.

        Parameters
        ----------
        update : Mapping
            The update as it was stored.

        Returns
        -------
        dict

        Raises
        ------
        StateStoreError
            If the update has a key that is not a state field.
        ResumeError
            If a stored value is not valid for its field.
        """
        self._check_stored_keys(update)
        restored: dict[str, Any] = {}
        for key, value in update.items():
            descriptor = self.fields.get(key)
            try:
                if _is_stored_replace(value):
                    inner = value[REPLACE_DELTA_KEY]
                    restored[key] = Replace(
                        inner
                        if descriptor is None
                        else _read(descriptor.adapter, inner)
                    )
                elif descriptor is None:
                    restored[key] = value
                elif descriptor.reducer is replace:
                    restored[key] = _read(descriptor.adapter, value)
                else:
                    restored[key] = _read_reduced(descriptor, value)
            except ValidationError as error:
                raise ResumeError(
                    f"The stored update of state field '{key}' is not valid for "
                    f"state schema {self._name}: {_describe(error)}"
                ) from error
        return restored


def _merge(
    schema: StateSchema[Any],
    current: dict[str, Any],
    update: dict[str, Any] | None,
    writers: Mapping[str, str],
) -> tuple[dict[str, Any], set[str]]:
    if not update:
        return current, set()
    next_state = dict(current)
    written: set[str] = set()
    for key, value in update.items():
        descriptor = schema.fields.get(key)
        writer = writers.get(key)
        if descriptor is None and not schema.is_dynamic():
            raise StateUpdateError(
                f"Unknown field '{key}' in state update{_source(writer)}"
            )
        if isinstance(value, Replace):
            next_state[key] = value.value
        elif descriptor is None or descriptor.reducer is replace:
            next_state[key] = value
        else:
            next_state[key] = _reduce(
                key, writer, descriptor, next_state.get(key), value
            )
        written.add(key)
    return next_state, written


def merge_updates(
    schema: StateSchema[Any],
    current: dict[str, Any],
    update: dict[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    """Apply an update to a state through the field reducers.

    The update comes from ``StateSchema.validate_update`` or
    ``StateSchema.restore_update``. Values of ``replace`` fields and
    ``Replace(value)`` are set as they are; for another reducer the merged
    value is validated when the field has its own constraints or validators,
    or a custom reducer.

    Parameters
    ----------
    schema : StateSchema
    current : dict
    update : dict or None

    Returns
    -------
    tuple[dict, set[str]]
        The new state and the written field names.

    Raises
    ------
    StateUpdateError
        If the update contains an unknown field, a reducer rejects a value, or
        a merged value is not valid for its field.
    """
    return _merge(schema, current, update, {})


def _detect_parallel_conflicts(
    schema: StateSchema[Any],
    updates: list[tuple[str, dict[str, Any] | None]],
) -> None:
    writers: dict[str, set[str]] = {}
    exclusive: set[str] = set()
    replaced: set[str] = set()
    for node_name, update in updates:
        if not update:
            continue
        for key, value in update.items():
            writers.setdefault(key, set()).add(node_name)
            is_replace = isinstance(value, Replace)
            descriptor = schema.fields.get(key)
            reducer = descriptor.reducer if descriptor is not None else replace
            if is_replace or reducer is replace:
                exclusive.add(key)
            if is_replace and reducer is not replace:
                replaced.add(key)
    for field_name, nodes in writers.items():
        if field_name in exclusive and len(nodes) > 1:
            raise InvalidUpdateError(
                field=field_name, nodes=list(nodes), replacement=field_name in replaced
            )


def merge_parallel_updates(
    schema: StateSchema[Any],
    current: dict[str, Any],
    updates: list[tuple[str, dict[str, Any] | None]],
) -> tuple[dict[str, Any], set[str]]:
    """Apply the updates of one superstep in node-name order.

    Each update is merged as in ``merge_updates``.

    Parameters
    ----------
    schema : StateSchema
    current : dict
    updates : list[tuple[str, dict or None]]
        Updates keyed by node name.

    Returns
    -------
    tuple[dict, set[str]]

    Raises
    ------
    InvalidUpdateError
        If different nodes write the same field and one of the writes has no
        merging reducer or is a ``Replace``.
    StateUpdateError
        If a reducer rejects a value or a merged value is not valid for its
        field; the error names every node that wrote the field.
    """
    _detect_parallel_conflicts(schema, updates)
    ordered = sorted(updates, key=lambda item: item[0])
    nodes: dict[str, list[str]] = {}
    for node_name, update in ordered:
        for key in update or {}:
            names = nodes.setdefault(key, [])
            if node_name not in names:
                names.append(node_name)
    writers = {
        key: f"node{'s' if len(names) > 1 else ''} "
        f"{', '.join(repr(name) for name in names)}"
        for key, names in nodes.items()
    }
    next_state = dict(current)
    written: set[str] = set()
    for _, update in ordered:
        next_state, step_written = _merge(schema, next_state, update, writers)
        written |= step_written
    return next_state, written


def snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a state dict."""
    return copy.deepcopy(state)


class StateSnapshot(NodestepModel, Generic[StateT]):
    """A thread's state (``value``) as of the history event numbered ``sequence``."""

    value: StateT
    sequence: int


def _appended(old: list[Any], new: list[Any]) -> list[Any] | None:
    if len(new) < len(old) or not all(
        _unchanged(old_item, new_item)
        for old_item, new_item in zip(old, new, strict=False)
    ):
        return None
    return new[len(old) :]


def _message_delta(old: list[Any], new: list[Any]) -> list[Any] | None:
    old_by_id = {message_id(message): message for message in old}
    if None in old_by_id:
        return _appended(old, new)
    if not set(old_by_id) <= {message_id(message) for message in new}:
        return None
    return [
        message
        for message in new
        if message_id(message) not in old_by_id
        or not _unchanged(old_by_id[message_id(message)], message)
    ]


def _list_delta(descriptor: FieldDescriptor, old: Any, new: Any) -> Any:
    old_list = [] if old is None else old
    if isinstance(old_list, list) and isinstance(new, list):
        delta = (
            _message_delta(old_list, new)
            if descriptor.reducer is add_messages
            else _appended(old_list, new)
        )
        if delta is not None:
            return delta
    return Replace(new)


def _dict_delta(old: Any, new: Any) -> Any:
    old_dict = {} if old is None else old
    if (
        isinstance(old_dict, dict)
        and isinstance(new, dict)
        and old_dict.keys() <= new.keys()
    ):
        return {
            key: value
            for key, value in new.items()
            if key not in old_dict or not _unchanged(old_dict[key], value)
        }
    return Replace(new)


def _diff_field(
    descriptor: FieldDescriptor | None,
    old_value: Any,
    new_value: Any,
) -> Any:
    if descriptor is None:
        return Replace(new_value)
    reducer = descriptor.reducer
    if reducer is replace:
        return new_value
    if reducer is add or reducer is add_messages:
        return _list_delta(descriptor, old_value, new_value)
    if reducer is merge_dict:
        return _dict_delta(old_value, new_value)
    raise GraphExecutionError(f"field '{descriptor.name}' has a custom reducer")


def _changed_fields(
    schema: StateSchema[Any], before: Any, after: Any
) -> Iterator[tuple[str, Any, Any]]:
    before_dict = schema.to_dict(before)
    after_dict = schema.to_dict(after)
    if schema.is_dynamic():
        keys: Iterable[str] = before_dict.keys() | after_dict.keys()
    else:
        keys = [
            *schema.fields,
            *(key for key in after_dict if key not in schema.fields),
        ]
    for key in keys:
        old_value = before_dict.get(key)
        new_value = after_dict.get(key)
        if (key in before_dict) != (key in after_dict) or not _unchanged(
            old_value, new_value
        ):
            yield key, old_value, new_value


def states_differ(schema: StateSchema[Any], before: Any, after: Any) -> bool:
    """Whether two states differ in their values, types or JSON forms.

    Fields are compared item by item through dicts (key order included),
    lists, tuples, deques, sets (in any order), dataclasses and pydantic
    models (fields, extra and private attributes). ``1`` and ``True``, a tuple
    and a list, objects of two classes with the same fields, and datetimes for
    one instant in different time zones all differ; an equal copy does not,
    and a field only one state has does. See
    [What a node returns](../../concepts/nodes.md#what-a-node-returns).

    Parameters
    ----------
    schema : StateSchema
    before : dict or BaseModel
    after : dict or BaseModel

    Returns
    -------
    bool
    """
    return next(_changed_fields(schema, before, after), None) is not None


def diff_states(
    schema: StateSchema,
    before: Any,
    after: Any,
) -> dict[str, Any]:
    """Compute the update that turns one state into another.

    Fields, list items and dict entries are compared as in
    ``states_differ``. ``add`` and ``add_messages`` fields yield only the
    appended or changed items, ``merge_dict`` fields only the added or changed
    keys, other fields their new value; a change these reducers cannot express
    becomes a ``Replace``. Keys of ``after`` that are not schema fields are
    included, so merging the update rejects them.

    Parameters
    ----------
    schema : StateSchema
    before : dict or BaseModel
    after : dict or BaseModel

    Returns
    -------
    dict

    Raises
    ------
    GraphExecutionError
        If a changed field has a reducer other than ``replace``, ``add``,
        ``add_messages`` and ``merge_dict``, whose update cannot be derived
        from the change.
    """
    return {
        key: _diff_field(schema.fields.get(key), old_value, new_value)
        for key, old_value, new_value in _changed_fields(schema, before, after)
    }


__all__ = [
    "FieldDescriptor",
    "StateSchema",
    "StateSnapshot",
    "diff_states",
    "merge_parallel_updates",
    "merge_updates",
    "snapshot_state",
    "states_differ",
]

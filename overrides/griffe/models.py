import ast
from typing import Any

from griffe import (
    Attribute,
    Class,
    Docstring,
    Expr,
    ExprCall,
    ExprKeyword,
    ExprLambda,
    Extension,
    Function,
    Module,
    Parameter,
    ParameterKind,
    Parameters,
)

MODEL_BASES = {"pydantic.BaseModel", "pydantic.main.BaseModel"}
SETTINGS_BASES = {
    "pydantic_settings.BaseSettings",
    "pydantic_settings.main.BaseSettings",
}
FIELD_FUNCTIONS = {"pydantic.Field", "pydantic.fields.Field"}
EMPTY_FACTORIES = {"list": "[]", "dict": "{}"}


class ModelField:
    def __init__(self, attribute: Attribute) -> None:
        self.attribute = attribute

    @classmethod
    def accepts(cls, member: Attribute) -> bool:
        return (
            member.annotation is not None
            and not member.name.startswith("_")
            and member.name != "model_config"
            and "instance-attribute" in member.labels
        )

    def call(self) -> ExprCall | None:
        value = self.attribute.value
        if isinstance(value, ExprCall) and value.canonical_path in FIELD_FUNCTIONS:
            return value
        return None

    def keyword(self, name: str) -> str | Expr | None:
        call = self.call()
        for argument in call.arguments if call else ():
            if isinstance(argument, ExprKeyword) and argument.name == name:
                return argument.value
        return None

    def default(self) -> str | Expr | None:
        call = self.call()
        if call is None:
            return self.attribute.value
        factory = self.keyword("default_factory")
        if factory is not None:
            return self.made_by(factory)
        default = self.keyword("default")
        positional = [
            argument
            for argument in call.arguments
            if not isinstance(argument, ExprKeyword)
        ]
        if default is None and positional:
            default = positional[0]
        return None if default == "..." else default

    @staticmethod
    def made_by(factory: str | Expr) -> str | Expr:
        if isinstance(factory, ExprLambda):
            return factory.body
        return EMPTY_FACTORIES.get(str(factory), f"{factory}()")

    def describe(self) -> None:
        description = self.keyword("description")
        if self.attribute.docstring is None and isinstance(description, str):
            self.attribute.docstring = Docstring(
                ast.literal_eval(description), parent=self.attribute
            )

    def parameter(self) -> Parameter:
        return Parameter(
            self.attribute.name,
            annotation=self.attribute.annotation,
            kind=ParameterKind.keyword_only,
            default=self.default(),
            docstring=self.attribute.docstring,
        )


class ModelClass:
    def __init__(self, cls: Class) -> None:
        self.cls = cls

    def lineage(self) -> list[Class]:
        return [*reversed(self.cls.mro()), self.cls]

    def bases(self) -> set[str]:
        return {
            base.canonical_path
            for owner in self.lineage()
            for base in owner.bases
            if isinstance(base, Expr)
        }

    def own_fields(self, owner: Class) -> list[ModelField]:
        return [
            ModelField(member)
            for member in owner.members.values()
            if isinstance(member, Attribute) and ModelField.accepts(member)
        ]

    def fields(self) -> list[ModelField]:
        found: dict[str, ModelField] = {}
        for owner in self.lineage():
            for field in self.own_fields(owner):
                found[field.attribute.name] = field
        return list(found.values())

    def apply(self) -> None:
        bases = self.bases()
        if not bases & (MODEL_BASES | SETTINGS_BASES):
            return
        for field in self.own_fields(self.cls):
            field.describe()
        if bases & SETTINGS_BASES or "__init__" in self.cls.members:
            return
        self.cls.set_member(
            "__init__",
            Function(
                "__init__",
                lineno=0,
                endlineno=0,
                parent=self.cls,
                parameters=Parameters(
                    Parameter("self", kind=ParameterKind.positional_or_keyword),
                    *(field.parameter() for field in self.fields()),
                ),
                returns="None",
            ),
        )


class PydanticModels(Extension):
    def on_package(self, *, pkg: Module, **kwargs: Any) -> None:
        self.walk(pkg, set())

    def walk(self, obj: Module | Class, seen: set[str]) -> None:
        if obj.canonical_path in seen:
            return
        seen.add(obj.canonical_path)
        if isinstance(obj, Class):
            ModelClass(obj).apply()
        for member in obj.members.values():
            if not member.is_alias and isinstance(member, Module | Class):
                self.walk(member, seen)

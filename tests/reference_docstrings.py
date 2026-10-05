import inspect
import pkgutil
import re
from collections.abc import Iterator
from pathlib import Path
from types import FunctionType

DIRECTIVE = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)
METHOD_TYPES = (FunctionType, classmethod, staticmethod)


class ReferenceDocstrings:
    def __init__(self, page: Path) -> None:
        self.page = page

    def __iter__(self) -> Iterator[tuple[str, str]]:
        text = self.page.read_text(encoding="utf-8")
        for identifier in DIRECTIVE.findall(text):
            value = pkgutil.resolve_name(identifier)
            yield identifier, inspect.getdoc(value) or ""
            if inspect.isclass(value):
                yield from self._methods(identifier, value)

    @staticmethod
    def _methods(identifier: str, owner: type) -> Iterator[tuple[str, str]]:
        for name, member in vars(owner).items():
            if not name.startswith("_") and isinstance(member, METHOD_TYPES):
                docstring = inspect.getdoc(getattr(owner, name)) or ""
                yield f"{identifier}.{name}", docstring

    def text(self) -> str:
        return "\n\n".join(docstring for _, docstring in self)

import importlib
import importlib.util
import inspect
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from nodestep import Graph
from nodestep_sandbox.errors import TargetError

_HIDDEN_DIRECTORIES = (Path(__file__).parent, Path(importlib.__file__).parent)


def _shown(filename: str) -> bool:
    return (
        not filename.startswith("<frozen importlib")
        and Path(filename).parent not in _HIDDEN_DIRECTORIES
    )


def _failure(action: str, error: Exception) -> TargetError:
    report = traceback.TracebackException.from_exception(error)
    report.stack = traceback.StackSummary.from_list(
        [frame for frame in report.stack if _shown(frame.filename)]
    )
    return TargetError(
        f"{action} failed: {type(error).__name__}: {error}",
        details="".join(report.format()),
    )


class ObjectPath(BaseModel):
    """Where an object lives: ``package.module:attribute`` or a file path.

    A file path is written ``path/to/file.py:attribute``.

    Attributes
    ----------
    source : str
        Module name, or path of a Python file.
    attribute : str
        Name of the object in that module.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    attribute: str

    @classmethod
    def parse(cls, text: str) -> Self:
        """Split ``text`` at its last colon.

        Parameters
        ----------
        text : str

        Returns
        -------
        ObjectPath

        Raises
        ------
        TargetError
            If there is no colon, nothing before it, or no identifier after it.
        """
        source, separator, attribute = text.rpartition(":")
        if not separator or not source or not attribute.isidentifier():
            raise TargetError(
                f"{text!r} is not package.module:attribute or path/to/file.py:attribute"
            )
        return cls(source=source, attribute=attribute)

    @property
    def is_file(self) -> bool:
        """Whether ``source`` is a file path rather than a module name."""
        return self.source.endswith(".py") or "/" in self.source or "\\" in self.source

    def __str__(self) -> str:
        return f"{self.source}:{self.attribute}"


class TargetLoader:
    """Loads the graph and the context named on the command line.

    ``cwd`` is put first on ``sys.path``, because console scripts do not add
    the current directory. A file target is imported under its file name
    without ``.py``.

    Parameters
    ----------
    cwd : Path
        Directory searched first for modules and used for relative file paths.
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    async def graph(self, text: str) -> Graph[Any]:
        """Load a ``Graph``, calling it first when it is a factory.

        Parameters
        ----------
        text : str
            ``package.module:attribute`` or ``path/to/file.py:attribute``;
            the attribute is a ``Graph`` or a function without arguments, sync
            or async, that returns one.

        Returns
        -------
        Graph

        Raises
        ------
        TargetError
            If the target is malformed, its module or file does not exist or
            raises when imported, the attribute is missing, is not a graph or a
            factory, the factory takes arguments, raises or returns something
            else, or the graph has no flow. When the loaded code raised,
            ``details`` holds its traceback without the sandbox's frames.
        """
        path = ObjectPath.parse(text)
        value = self._load(path)
        if not isinstance(value, Graph):
            if not callable(value):
                raise TargetError(
                    f"{path} is of type {type(value).__name__}, not a Graph or a "
                    "function that returns one"
                )
            value = await self._call(path, value)
            if not isinstance(value, Graph):
                raise TargetError(
                    f"{path} returned a value of type {type(value).__name__}, "
                    "not a Graph"
                )
        if value.start_node is None:
            raise TargetError(
                f"Graph '{value.name}' from {path} has no flow; call graph.flow(...)"
            )
        return value

    async def context(self, text: str) -> Any:
        """Load the run context: an object, or the result of a factory.

        Parameters
        ----------
        text : str
            Same form as for ``graph``. A callable (a function or a class) is
            called once without arguments, and awaited when that gives an
            awaitable; anything else is the context as it is.

        Returns
        -------
        Any

        Raises
        ------
        TargetError
            If the target cannot be loaded, or the factory takes arguments or
            raises.
        """
        path = ObjectPath.parse(text)
        value = self._load(path)
        if callable(value):
            return await self._call(path, value)
        return value

    def _load(self, path: ObjectPath) -> Any:
        directory = str(self.cwd)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        module = self._load_file(path) if path.is_file else self._import(path)
        if not hasattr(module, path.attribute):
            raise TargetError(f"{path.source} has no attribute {path.attribute!r}")
        return getattr(module, path.attribute)

    def _import(self, path: ObjectPath) -> ModuleType:
        try:
            return importlib.import_module(path.source)
        except Exception as error:
            missing = error.name if isinstance(error, ModuleNotFoundError) else None
            if missing and (
                path.source == missing or path.source.startswith(f"{missing}.")
            ):
                raise TargetError(
                    f"No module named {missing!r}; start the sandbox in the directory "
                    "that contains it, or install it"
                ) from error
            raise _failure(f"Importing {path.source}", error) from error

    def _load_file(self, path: ObjectPath) -> ModuleType:
        file = (self.cwd / path.source).resolve()
        if not file.is_file():
            raise TargetError(f"File {file} does not exist")
        name = file.stem
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is not None and Path(origin).resolve() == file:
                return loaded
            raise TargetError(
                f"A module named {name!r} is already imported from somewhere else; "
                f"rename {file.name} or load it as package.module:{path.attribute}"
            )
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise TargetError(f"{file} cannot be imported as a Python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            del sys.modules[name]
            raise _failure(f"Importing {file.name}", error) from error
        except BaseException:
            del sys.modules[name]
            raise
        return module

    async def _call(self, path: ObjectPath, factory: Callable[..., Any]) -> Any:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            required = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.default is parameter.empty
                and parameter.kind
                not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            ]
            if required:
                raise TargetError(
                    f"{path} takes arguments ({', '.join(required)}); the sandbox "
                    "calls it without any"
                )
        try:
            result = factory()
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            raise _failure(f"Calling {path}", error) from error
        return result

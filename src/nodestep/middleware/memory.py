"""Long-term memory for the model, stored as JSON files by `FilesystemMemory`.

See [Memory](../../concepts/workspace.md#memory).
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from pydantic import Field

from nodestep.core.tool import Tool, ToolContext, tool
from nodestep.exceptions import NodestepError, PathAccessError
from nodestep.middleware.base import Middleware
from nodestep.models.base import NodestepModel
from nodestep.workspace.paths import check_path


class BaseMemory(Middleware, ABC):
    """Base class for middleware that gives the model long-term memory."""

    @abstractmethod
    def tools(self) -> Iterable[Tool]:
        """Return the memory tools."""
        ...


class Memory(NodestepModel):
    """A stored memory entry.

    Attributes
    ----------
    key : str
        Name of the entry within its scope; saving the same key replaces it.
    memory_type : str
        Free label chosen by the model, such as ``"fact"``.
    scope : str
        Scope such as ``"global"`` or ``"user:42"``, set by the middleware.
    created_at : datetime or None
        Set to the current time when not given; a save keeps the stored one.
    updated_at : datetime or None
        Set to the current time when not given and on every save.
    """

    key: str
    title: str
    content: str
    memory_type: str = "fact"
    scope: str = "global"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def model_post_init(self, __context: object) -> None:
        now = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def searchable_text(self) -> str:
        """Text used for keyword search."""
        return f"{self.title} {self.content}"


class MemorySearchResult(NodestepModel):
    """Result of the ``search_memory`` tool.

    Attributes
    ----------
    results : list[Memory]
        Matching memories, best first.
    skipped : list[str]
        Memory files that could not be read.
    """

    query: str
    results: list[Memory]
    skipped: list[str] = Field(default_factory=list)


class MemoryListResult(NodestepModel):
    """Memories of one scope, the result of the ``list_memories`` tool.

    Attributes
    ----------
    skipped : list[str]
        Memory files that could not be read.
    """

    scope: str
    entries: list[Memory]
    skipped: list[str] = Field(default_factory=list)


class MemoryDeleteResult(NodestepModel):
    """Result of the ``delete_memory`` tool.

    Attributes
    ----------
    deleted : bool
        Whether an entry with the key existed and was deleted.
    """

    key: str
    scope: str
    deleted: bool


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_search(
    memories: list[Memory],
    query: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[Memory]:
    """Rank memories against a query with BM25 keyword scoring.

    Parameters
    ----------
    memories : list[Memory]
    query : str
        Words to look for in the titles and contents.
    top_k : int, optional
        Maximum number of results.
    k1 : float, optional
        BM25 term saturation: how much repeated words count.
    b : float, optional
        BM25 length normalization, from 0 (none) to 1 (full).

    Returns
    -------
    list[Memory]
        Matching memories, best first.

    Raises
    ------
    ValueError
        If ``top_k`` is smaller than 1.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not memories or not query.strip():
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    documents = [_tokenize(memory.searchable_text()) for memory in memories]
    document_count = len(documents)
    average_length = (
        sum(len(document) for document in documents) / document_count
        if document_count > 0
        else 1.0
    )

    document_frequency: dict[str, int] = {}
    for document in documents:
        for term in set(document):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scores: list[tuple[float, int]] = []
    for index, document in enumerate(documents):
        term_counts: dict[str, int] = {}
        for term in document:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        document_length = len(document)
        for term in query_terms:
            if term not in document_frequency:
                continue
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            frequency = document_frequency[term]
            inverse_frequency = math.log(
                (document_count - frequency + 0.5) / (frequency + 0.5) + 1.0
            )
            numerator = term_frequency * (k1 + 1.0)
            denominator = term_frequency + k1 * (
                1.0 - b + b * document_length / average_length
            )
            score += inverse_frequency * numerator / denominator

        if score > 0:
            scores.append((score, index))

    scores.sort(key=lambda scored: -scored[0])
    return [memories[index] for _, index in scores[:top_k]]


class MemoryConfigError(NodestepError):
    """Raised when ``FilesystemMemory`` is configured with an unusable root or scope."""


def _memories_dir(root_dir: Path, scope: str) -> Path:
    root = root_dir / "memories"
    segments = scope.split(":")
    if any(segment in ("", ".", "..") for segment in segments):
        raise PathAccessError(f"Invalid memory scope '{scope}'")
    safe_scope = "/".join(quote(part, safe="") for part in segments)
    check_path(safe_scope)
    candidate = (root / safe_scope).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PathAccessError(
            f"Memory scope '{scope}' resolves outside the memories root"
        ) from error
    return candidate


def _memory_path(root_dir: Path, scope: str, key: str) -> Path:
    file_name = f"{quote(key, safe='')}.json"
    check_path(file_name)
    return _memories_dir(root_dir, scope) / file_name


def save_memory_to_disk(root_dir: Path, memory: Memory) -> Memory:
    """Write a memory as JSON, keeping its original creation time.

    Parameters
    ----------
    root_dir : Path
        Folder that holds the ``memories`` folder.
    memory : Memory

    Returns
    -------
    Memory

    Raises
    ------
    ValueError
        If a file for the key exists but is not valid UTF-8 memory JSON; the
        file is left as it is.
    """
    memory.updated_at = datetime.now(UTC)
    path = _memory_path(root_dir, memory.scope, memory.key)
    if path.exists():
        try:
            existing = Memory.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(
                f"Memory file {path.name} of scope {memory.scope!r} is not valid "
                "UTF-8 memory JSON; move or delete it before saving this key"
            ) from error
        memory.created_at = existing.created_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    return memory


def load_memories_from_disk(root_dir: Path, scope: str) -> MemoryListResult:
    """Load all memories of a scope.

    Parameters
    ----------
    root_dir : Path
        Folder that holds the ``memories`` folder.
    scope : str
        Scope such as ``"global"`` or ``"user:42"``.

    Returns
    -------
    MemoryListResult
        The memories, and in ``skipped`` the names of files that are not valid
        UTF-8 memory JSON.
    """
    directory = _memories_dir(root_dir, scope)
    result = MemoryListResult(scope=scope, entries=[])
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        try:
            result.entries.append(
                Memory.model_validate_json(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            result.skipped.append(path.name)
    return result


def delete_memory_from_disk(root_dir: Path, scope: str, key: str) -> bool:
    """Delete one memory.

    Parameters
    ----------
    root_dir : Path
        Folder that holds the ``memories`` folder.
    scope : str
    key : str

    Returns
    -------
    bool
        Whether a memory was deleted.
    """
    path = _memory_path(root_dir, scope, key)
    if path.exists():
        path.unlink()
        return True
    return False


class FilesystemMemory(BaseMemory):
    """Memory tools that store entries as UTF-8 JSON files under ``root_dir``.

    Entries of scope ``"user:42"`` live in ``root_dir/memories/user/42/``.

    Parameters
    ----------
    root_dir : str or Path
        Storage directory; created on the first write.
    scope : str or callable
        The scope, or a function of the call's ``ToolContext`` that returns it,
        e.g. ``lambda ctx: f"user:{ctx.context.user_id}"``. Segments are
        separated by ``:``; empty, ``.`` and ``..`` segments are refused.

    Raises
    ------
    MemoryConfigError
        If ``root_dir`` is an existing file or ``scope`` is an invalid string.
    TypeError
        If ``scope`` is neither a string nor callable.
    """

    def __init__(
        self, root_dir: str | Path, *, scope: str | Callable[[ToolContext], str]
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        if self.root_dir.is_file():
            raise MemoryConfigError(
                f"FilesystemMemory root_dir {self.root_dir} is a file, not a directory"
            )
        if isinstance(scope, str):
            try:
                _memories_dir(self.root_dir, scope)
            except PathAccessError as error:
                raise MemoryConfigError(
                    f"FilesystemMemory scope {scope!r} is invalid: {error}"
                ) from error
        elif not callable(scope):
            raise TypeError(
                "FilesystemMemory scope must be a str or a callable taking the "
                f"ToolContext, got {type(scope).__name__}"
            )
        self.scope = scope
        self._tools = (
            self._build_save_memory(),
            self._build_search_memory(),
            self._build_list_memories(),
            self._build_delete_memory(),
        )

    def scope_for(self, ctx: ToolContext) -> str:
        """Return the scope of a tool call.

        Parameters
        ----------
        ctx : ToolContext

        Returns
        -------
        str

        Raises
        ------
        TypeError
            If the scope callable returns something other than a ``str``.
        """
        if isinstance(self.scope, str):
            return self.scope
        scope = self.scope(ctx)
        if not isinstance(scope, str):
            raise TypeError(
                f"FilesystemMemory scope callable returned {type(scope).__name__}, "
                "expected str"
            )
        return scope

    def _build_save_memory(self) -> Tool:
        middleware = self

        @tool(
            name="save_memory", description="Save a memory entry to persistent storage."
        )
        async def save_memory(
            ctx: ToolContext,
            key: str,
            title: str,
            content: str,
            memory_type: str = "fact",
        ) -> Memory:
            memory = Memory(
                key=key,
                title=title,
                content=content,
                memory_type=memory_type,
                scope=middleware.scope_for(ctx),
            )
            return save_memory_to_disk(middleware.root_dir, memory)

        return save_memory

    def _build_search_memory(self) -> Tool:
        middleware = self

        @tool(
            name="search_memory", description="Search memories using BM25 text search."
        )
        async def search_memory(
            ctx: ToolContext,
            query: str,
            top_k: Annotated[int, Field(ge=1)] = 10,
        ) -> MemorySearchResult:
            loaded = load_memories_from_disk(
                middleware.root_dir, middleware.scope_for(ctx)
            )
            return MemorySearchResult(
                query=query,
                results=bm25_search(loaded.entries, query, top_k=top_k),
                skipped=loaded.skipped,
            )

        return search_memory

    def _build_list_memories(self) -> Tool:
        middleware = self

        @tool(name="list_memories", description="List all memory entries.")
        async def list_memories(ctx: ToolContext) -> MemoryListResult:
            return load_memories_from_disk(
                middleware.root_dir, middleware.scope_for(ctx)
            )

        return list_memories

    def _build_delete_memory(self) -> Tool:
        middleware = self

        @tool(name="delete_memory", description="Delete a memory entry.")
        async def delete_memory(ctx: ToolContext, key: str) -> MemoryDeleteResult:
            scope = middleware.scope_for(ctx)
            deleted = delete_memory_from_disk(middleware.root_dir, scope, key)
            return MemoryDeleteResult(key=key, scope=scope, deleted=deleted)

        return delete_memory

    def tools(self) -> tuple[Tool, ...]:
        """Return the save, search, list and delete tools."""
        return self._tools


__all__ = [
    "BaseMemory",
    "FilesystemMemory",
    "Memory",
    "MemoryConfigError",
    "MemoryDeleteResult",
    "MemoryListResult",
    "MemorySearchResult",
    "bm25_search",
    "delete_memory_from_disk",
    "load_memories_from_disk",
    "save_memory_to_disk",
]

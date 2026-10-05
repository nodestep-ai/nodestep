from pathlib import Path
from typing import Any

import pytest

from nodestep.core.tool import ToolContext, call_tool
from nodestep.exceptions import PathAccessError
from nodestep.middleware.memory import (
    FilesystemMemory,
    Memory,
    MemoryConfigError,
    bm25_search,
    delete_memory_from_disk,
    load_memories_from_disk,
    save_memory_to_disk,
)


def _tools(memory: FilesystemMemory) -> dict[str, Any]:
    return {tool.name: tool for tool in memory.tools()}


def test_save_and_load_memory(tmp_path: Path) -> None:
    memory = Memory(
        key="pref-lang", title="Language preference", content="User prefers Spanish"
    )
    save_memory_to_disk(tmp_path, memory)
    loaded = load_memories_from_disk(tmp_path, "global").entries
    assert len(loaded) == 1
    assert loaded[0].key == "pref-lang"
    assert loaded[0].content == "User prefers Spanish"


def test_delete_memory(tmp_path: Path) -> None:
    save_memory_to_disk(tmp_path, Memory(key="tmp", title="temp", content="data"))
    assert delete_memory_from_disk(tmp_path, "global", "tmp") is True
    assert delete_memory_from_disk(tmp_path, "global", "tmp") is False
    assert load_memories_from_disk(tmp_path, "global").entries == []


def test_scoped_memories_are_isolated(tmp_path: Path) -> None:
    save_memory_to_disk(
        tmp_path, Memory(key="a", title="A", content="global", scope="global")
    )
    save_memory_to_disk(
        tmp_path, Memory(key="b", title="B", content="user", scope="user:123")
    )
    assert len(load_memories_from_disk(tmp_path, "global").entries) == 1
    assert len(load_memories_from_disk(tmp_path, "user:123").entries) == 1


def test_bm25_search_ranks_by_relevance() -> None:
    memories = [
        Memory(
            key="a", title="Python tips", content="Use list comprehensions for speed"
        ),
        Memory(key="b", title="Java tips", content="Use streams for functional style"),
        Memory(
            key="c", title="Python patterns", content="Python decorators are powerful"
        ),
    ]
    results = bm25_search(memories, "python")
    assert len(results) == 2
    keys = [result.key for result in results]
    assert "a" in keys
    assert "c" in keys
    assert "b" not in keys


def test_memory_keys_with_special_chars_do_not_collide(tmp_path: Path) -> None:
    save_memory_to_disk(tmp_path, Memory(key="hello world", title="A", content="first"))
    save_memory_to_disk(
        tmp_path, Memory(key="hello@world", title="B", content="second")
    )
    loaded = load_memories_from_disk(tmp_path, "global").entries
    assert len(loaded) == 2
    assert {memory.key for memory in loaded} == {"hello world", "hello@world"}


def test_bm25_search_empty_query() -> None:
    memories = [Memory(key="a", title="test", content="hello")]
    assert bm25_search(memories, "") == []
    assert bm25_search(memories, "   ") == []


def test_bm25_search_no_match() -> None:
    memories = [Memory(key="a", title="test", content="hello world")]
    assert bm25_search(memories, "zzzzz") == []


async def test_save_memory_tool_uses_the_bound_scope(tmp_path: Path) -> None:
    memory = FilesystemMemory(tmp_path, scope="user:1")
    result: Any = (
        await call_tool(
            _tools(memory)["save_memory"],
            {"key": "lang", "title": "Language", "content": "Spanish"},
            ToolContext(),
        )
    ).value
    assert result.key == "lang"
    assert result.scope == "user:1"
    assert len(load_memories_from_disk(tmp_path, "user:1").entries) == 1


async def test_search_memory_tool(tmp_path: Path) -> None:
    tools = _tools(FilesystemMemory(tmp_path, scope="global"))
    ctx = ToolContext()
    await call_tool(
        tools["save_memory"],
        {"key": "a", "title": "Python", "content": "Use decorators"},
        ctx,
    )
    await call_tool(
        tools["save_memory"],
        {"key": "b", "title": "Java", "content": "Use streams"},
        ctx,
    )
    result: Any = (
        await call_tool(tools["search_memory"], {"query": "python"}, ctx)
    ).value
    assert result.query == "python"
    assert [memory.key for memory in result.results] == ["a"]
    assert result.skipped == []


async def test_list_memories_tool(tmp_path: Path) -> None:
    tools = _tools(FilesystemMemory(tmp_path, scope="global"))
    ctx = ToolContext()
    await call_tool(
        tools["save_memory"], {"key": "x", "title": "X", "content": "data"}, ctx
    )
    await call_tool(
        tools["save_memory"], {"key": "y", "title": "Y", "content": "more"}, ctx
    )
    result: Any = (await call_tool(tools["list_memories"], {}, ctx)).value
    assert len(result.entries) == 2
    assert result.scope == "global"


async def test_delete_memory_tool(tmp_path: Path) -> None:
    tools = _tools(FilesystemMemory(tmp_path, scope="global"))
    ctx = ToolContext()
    await call_tool(
        tools["save_memory"], {"key": "z", "title": "Z", "content": "temp"}, ctx
    )
    result: Any = (await call_tool(tools["delete_memory"], {"key": "z"}, ctx)).value
    assert result.deleted is True
    result2: Any = (await call_tool(tools["delete_memory"], {"key": "z"}, ctx)).value
    assert result2.deleted is False


def test_the_model_cannot_choose_the_scope(tmp_path: Path) -> None:
    for memory_tool in FilesystemMemory(tmp_path, scope="global").tools():
        assert "scope" not in memory_tool.input_schema()["properties"]


async def test_a_scope_argument_from_the_model_is_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    tools = _tools(FilesystemMemory(tmp_path, scope=lambda ctx: f"user:{ctx.run_id}"))

    with pytest.raises(ValidationError):
        await call_tool(
            tools["list_memories"], {"scope": "user:alice"}, ToolContext(run_id="bob")
        )


async def test_a_scope_callable_isolates_users(tmp_path: Path) -> None:
    tools = _tools(
        FilesystemMemory(tmp_path, scope=lambda ctx: f"user:{ctx.context['user']}")
    )
    alice = ToolContext(_context={"user": "alice"})
    bob = ToolContext(_context={"user": "bob"})

    await call_tool(
        tools["save_memory"], {"key": "ssn", "title": "SSN", "content": "123"}, alice
    )
    seen_by_bob: Any = (await call_tool(tools["list_memories"], {}, bob)).value
    seen_by_alice: Any = (await call_tool(tools["list_memories"], {}, alice)).value

    assert seen_by_bob.entries == []
    assert seen_by_bob.scope == "user:bob"
    assert [entry.key for entry in seen_by_alice.entries] == ["ssn"]
    assert (tmp_path / "memories" / "user" / "alice" / "ssn.json").exists()


async def test_a_scope_callable_must_return_a_string(tmp_path: Path) -> None:
    def scope(ctx: ToolContext) -> Any:
        return 42

    tools = _tools(FilesystemMemory(tmp_path, scope=scope))

    with pytest.raises(TypeError, match="scope"):
        await call_tool(tools["list_memories"], {}, ToolContext())


@pytest.mark.parametrize("scope", ["", "..", "team::x", "a:.:b"])
def test_an_invalid_scope_string_raises_at_construction(
    tmp_path: Path, scope: str
) -> None:
    with pytest.raises(MemoryConfigError, match="scope"):
        FilesystemMemory(tmp_path, scope=scope)


def test_scope_must_be_a_string_or_callable(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="scope"):
        FilesystemMemory(tmp_path, scope=42)  # ty: ignore[invalid-argument-type]


def test_root_dir_and_scope_are_required() -> None:
    with pytest.raises(TypeError):
        FilesystemMemory()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        FilesystemMemory("memories")  # ty: ignore[missing-argument]


def test_a_root_dir_that_is_a_file_raises_at_construction(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(MemoryConfigError, match="root_dir"):
        FilesystemMemory(target, scope="global")


async def test_unreadable_memory_files_are_reported_as_skipped(tmp_path: Path) -> None:
    tools = _tools(FilesystemMemory(tmp_path, scope="global"))
    ctx = ToolContext()
    await call_tool(
        tools["save_memory"], {"key": "good", "title": "Python", "content": "ok"}, ctx
    )
    folder = tmp_path / "memories" / "global"
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    (folder / "latin.json").write_bytes(b'{"key": "\xe9"}')
    (folder / "wrong.json").write_text('{"key": 1}', encoding="utf-8")

    listed: Any = (await call_tool(tools["list_memories"], {}, ctx)).value
    found: Any = (
        await call_tool(tools["search_memory"], {"query": "python"}, ctx)
    ).value

    assert [entry.key for entry in listed.entries] == ["good"]
    assert listed.skipped == ["broken.json", "latin.json", "wrong.json"]
    assert found.skipped == ["broken.json", "latin.json", "wrong.json"]
    assert [result.key for result in found.results] == ["good"]


def test_memory_files_are_read_and_written_as_utf8(tmp_path: Path) -> None:
    save_memory_to_disk(tmp_path, Memory(key="k", title="Café", content="naïve ☕"))

    raw = next((tmp_path / "memories" / "global").glob("*.json")).read_bytes()

    assert "naïve ☕".encode() in raw
    assert load_memories_from_disk(tmp_path, "global").entries[0].content == (
        "naïve ☕"
    )


def test_memory_scope_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathAccessError):
        save_memory_to_disk(
            tmp_path,
            Memory(key="x", title="A", content="data", scope="..:..:etc"),
        )


def test_memory_scope_traversal_rejected_on_load(tmp_path: Path) -> None:
    with pytest.raises(PathAccessError):
        load_memories_from_disk(tmp_path, "..:escape")


def test_memory_scope_traversal_rejected_on_delete(tmp_path: Path) -> None:
    with pytest.raises(PathAccessError):
        delete_memory_from_disk(tmp_path, "..:escape", "x")


def test_memory_nested_scope_still_isolated(tmp_path: Path) -> None:
    save_memory_to_disk(
        tmp_path, Memory(key="a", title="A", content="user", scope="user:123")
    )
    save_memory_to_disk(
        tmp_path, Memory(key="b", title="B", content="org", scope="org:9")
    )
    assert len(load_memories_from_disk(tmp_path, "user:123").entries) == 1
    assert len(load_memories_from_disk(tmp_path, "org:9").entries) == 1
    assert load_memories_from_disk(tmp_path, "user:999").entries == []


async def test_the_workspace_is_never_used_as_storage(tmp_path: Path) -> None:
    from nodestep.workspace import LocalWorkspace

    memory_root = tmp_path / "memories-home"
    workspace = LocalWorkspace(tmp_path / "agent-workspace", create=True)
    memory = FilesystemMemory(memory_root, scope="global")

    await call_tool(
        _tools(memory)["save_memory"],
        {"key": "lang", "title": "Language", "content": "Spanish"},
        ToolContext(workspace=workspace),
    )

    stored = load_memories_from_disk(memory_root.resolve(), "global").entries
    assert stored[0].key == "lang"
    assert not (workspace.root_dir / "memories").exists()


async def test_updating_a_memory_keeps_its_creation_time(tmp_path: Path) -> None:
    tools = _tools(FilesystemMemory(tmp_path, scope="global"))
    first: Any = (
        await call_tool(
            tools["save_memory"],
            {"key": "k", "title": "t", "content": "v1"},
            ToolContext(),
        )
    ).value
    second: Any = (
        await call_tool(
            tools["save_memory"],
            {"key": "k", "title": "t", "content": "v2"},
            ToolContext(),
        )
    ).value

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert load_memories_from_disk(tmp_path, "global").entries[0].content == "v2"


async def test_non_positive_top_k_is_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    tools = _tools(FilesystemMemory(tmp_path, scope="global"))

    with pytest.raises(ValidationError):
        await call_tool(
            tools["search_memory"], {"query": "x", "top_k": -1}, ToolContext()
        )


@pytest.mark.parametrize("scope", ["", ".", "..", "team:..", "team:.", "team::x"])
def test_scope_segments_that_alias_other_scopes_are_rejected(
    tmp_path: Path, scope: str
) -> None:
    with pytest.raises(PathAccessError):
        save_memory_to_disk(
            tmp_path, Memory(key="k", title="t", content="c", scope=scope)
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_bm25_search_rejects_non_positive_top_k(top_k: int) -> None:
    memories = [Memory(key="a", title="python", content="python tips")]

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        bm25_search(memories, "python", top_k=top_k)


@pytest.mark.parametrize(
    ("scope", "key"),
    [
        ("global", "nul"),
        ("global", "CON"),
        ("global", "com1"),
        ("user:CON", "x"),
        ("user:name.", "x"),
        ("aux", "x"),
    ],
)
def test_windows_reserved_keys_and_scopes_are_rejected(
    tmp_path: Path, scope: str, key: str
) -> None:
    with pytest.raises(PathAccessError):
        save_memory_to_disk(
            tmp_path, Memory(key=key, title="A", content="data", scope=scope)
        )
    with pytest.raises(PathAccessError):
        delete_memory_from_disk(tmp_path, scope, key)

    assert not (tmp_path / "memories").exists() or not any(
        (tmp_path / "memories").rglob("*.json")
    )


@pytest.mark.parametrize(
    "content", [b"{not json", b'{"key": "\xe9"}', b'{"key": 1}'], ids=str
)
def test_saving_over_an_unreadable_memory_file_raises(
    tmp_path: Path, content: bytes
) -> None:
    folder = tmp_path / "memories" / "global"
    folder.mkdir(parents=True)
    (folder / "k.json").write_bytes(content)

    with pytest.raises(ValueError, match=r"k\.json"):
        save_memory_to_disk(tmp_path, Memory(key="k", title="A", content="new"))

    assert (folder / "k.json").read_bytes() == content


def test_root_dir_is_not_expanded_with_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    memory = FilesystemMemory("~/notes", scope="global")

    assert memory.root_dir == tmp_path / "~" / "notes"

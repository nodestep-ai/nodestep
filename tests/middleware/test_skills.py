from pathlib import Path

import pytest

from nodestep import END, START, Graph, ScriptedChat, node
from nodestep.chat import HumanMessage, SystemMessage
from nodestep.core import model_node
from nodestep.core.builtin_nodes.agent import AgentState
from nodestep.middleware import FilesystemSkills, SkillCatalog, SkillFormatError


def _write(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_name_and_description_come_from_the_frontmatter(tmp_path: Path) -> None:
    _write(
        tmp_path / "pdf-tools",
        "---\n"
        "name: pdf-extractor\n"
        'description: "Extract text: tables and forms"\n'
        "license: MIT\n"
        "---\n"
        "# PDF Tools\n"
        "```bash\n#!/usr/bin/env bash\n```\n",
    )

    [skill] = SkillCatalog([tmp_path]).list_skills()

    assert skill.name == "pdf-extractor"
    assert skill.description == "Extract text: tables and forms"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("# Deploy\n", "frontmatter"),
        ("---\nname: deploy\n", "closing"),
        ("---\ndescription: Deploy it\n---\n", "name"),
        ("---\nname: deploy\n---\n", "description"),
        ("---\nname: deploy\ndescription:\n---\n", "description"),
        ("---\nname: deploy\nname: other\ndescription: d\n---\n", "twice"),
        ("---\nname: deploy\nnot a pair\ndescription: d\n---\n", "key: value"),
    ],
)
def test_invalid_frontmatter_raises(tmp_path: Path, text: str, problem: str) -> None:
    path = _write(tmp_path / "deploy", text)

    with pytest.raises(SkillFormatError, match=problem) as info:
        SkillCatalog([tmp_path])

    assert str(path) in str(info.value)


def test_a_skill_file_that_is_not_utf8_raises_with_its_path(tmp_path: Path) -> None:
    path = tmp_path / "deploy" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes(b"---\nname: deploy\ndescription: caf\xe9\n---\n")

    with pytest.raises(SkillFormatError, match="not valid UTF-8") as info:
        SkillCatalog([tmp_path])

    assert str(path) in str(info.value)


def test_a_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="skils"):
        SkillCatalog([tmp_path / "skils"])


def test_a_root_without_skill_files_raises(tmp_path: Path) -> None:
    (tmp_path / "skills" / "research").mkdir(parents=True)
    (tmp_path / "skills" / "research" / "notes.md").write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"no SKILL\.md"):
        SkillCatalog([tmp_path / "skills"])


def test_skill_files_are_read_as_utf8(tmp_path: Path) -> None:
    path = tmp_path / "cafe" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes("---\nname: café\ndescription: Crème ☕\n---\n".encode())

    [skill] = SkillCatalog([tmp_path]).list_skills()

    assert (skill.name, skill.description) == ("café", "Crème ☕")


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    _write(tmp_path / "research", "---\nname: research\ndescription: Dig in\n---\n")
    return tmp_path


def test_nodes_must_be_a_collection_of_names(skills_root: Path) -> None:
    with pytest.raises(TypeError, match="nodes"):
        FilesystemSkills([skills_root], nodes="think")
    with pytest.raises(ValueError, match="nodes"):
        FilesystemSkills([skills_root], nodes=())


async def test_the_catalog_reaches_only_the_listed_model_nodes(
    skills_root: Path,
) -> None:
    seen_by_plain_node: list[list] = []

    @node
    def prepare(state: AgentState) -> dict:
        seen_by_plain_node.append(list(state.messages))
        return {}

    think_chat = ScriptedChat(["thought"])
    answer_chat = ScriptedChat(["answer"])
    think = model_node("think", chat=think_chat, system_prompt="SYS")
    answer = model_node("answer", chat=answer_chat)
    skills = FilesystemSkills([skills_root], nodes={"think"})
    graph = Graph(AgentState, middleware=[skills]).flow(
        START >> prepare, prepare >> think, think >> answer, answer >> END
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})

    catalog = SystemMessage(
        content="Available skills (call load_skill(name) to expand):\n"
        "- research: Dig in"
    )
    assert think_chat.requests[0].messages == [
        SystemMessage(content="SYS"),
        catalog,
        HumanMessage(content="hi", id=result.data["messages"][0].id),
    ]
    assert catalog not in answer_chat.requests[0].messages
    assert [type(message) for message in seen_by_plain_node[0]] == [HumanMessage]
    assert all(
        message.content != catalog.content for message in result.data["messages"]
    )


def test_a_skill_file_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    path = tmp_path / "deploy" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes(
        "---\nname: deploy\ndescription: Ship it\n---\n".encode("utf-8-sig")
    )

    [skill] = SkillCatalog([tmp_path]).list_skills()

    assert (skill.name, skill.description) == ("deploy", "Ship it")


def test_a_root_file_that_is_not_a_skill_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# notes\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"README\.md"):
        SkillCatalog([path])

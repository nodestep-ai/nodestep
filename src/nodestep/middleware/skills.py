"""Skills: `SKILL.md` files that `FilesystemSkills` offers the model to load.

See [Built-in middleware](../../concepts/middleware.md#built-in-middleware).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from nodestep.chat import SystemMessage
from nodestep.core.tool import Tool, tool
from nodestep.exceptions import NodestepError
from nodestep.middleware.base import Middleware, ModelMiddlewareContext, Replacement


class BaseSkills(Middleware, ABC):
    """Base class for middleware that offers skills to the model."""

    @abstractmethod
    def tools(self) -> Iterable[Tool]:
        """Return the skill tools."""
        ...


class SkillNotFoundError(NodestepError, LookupError):
    """Raised when a skill name is unknown."""


class SkillConflictError(NodestepError):
    """Raised when two skill folders share a name."""


class SkillFormatError(NodestepError):
    """Raised when a ``SKILL.md`` file is not UTF-8 or has no valid frontmatter."""


class Skill(BaseModel):
    """A skill loaded from a ``SKILL.md`` file.

    Attributes
    ----------
    name : str
        The frontmatter ``name``.
    description : str
        The frontmatter ``description``, shown in the skill list.
    path : Path
        The ``SKILL.md`` file.
    root_path : Path
        The folder that holds it.
    content : str
        The whole file, frontmatter included.
    """

    name: str
    description: str
    path: Path
    root_path: Path
    content: str


class LoadedSkill(BaseModel):
    """Result of the ``load_skill`` tool.

    Attributes
    ----------
    content : str
        The whole ``SKILL.md``, frontmatter included.
    path : str
        The skill's folder.
    primary_file : str
        Its ``SKILL.md`` file.
    """

    name: str
    description: str
    content: str
    path: str
    primary_file: str


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _read_frontmatter(path: Path, content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError(
            f"{path} must start with a '---' frontmatter block holding name and "
            "description"
        )
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not key:
            raise SkillFormatError(
                f"{path}: frontmatter line {line!r} is not 'key: value'"
            )
        if key in fields:
            raise SkillFormatError(f"{path}: frontmatter key '{key}' is given twice")
        fields[key] = _unquote(value.strip())
    else:
        raise SkillFormatError(f"{path}: the frontmatter has no closing '---' line")
    for required in ("name", "description"):
        if not fields.get(required):
            raise SkillFormatError(
                f"{path}: the frontmatter needs a non-empty '{required}'"
            )
    return fields


def _is_skill_file(path: Path) -> bool:
    return path.is_file() and path.name == "SKILL.md"


class SkillCatalog:
    """Skills read from ``SKILL.md`` files when the catalog is built.

    A ``SKILL.md`` starts with ``key: value`` lines between two ``---`` lines:
    ``name`` and ``description`` are required, other keys are ignored, values
    may be quoted. Files are read as UTF-8, with or without a byte order mark.

    Parameters
    ----------
    roots : Iterable[str or Path]
        Directories searched recursively for ``SKILL.md`` files, or skill
        files themselves.

    Raises
    ------
    FileNotFoundError
        If a root does not exist or holds no ``SKILL.md``.
    ValueError
        If a root is a file other than ``SKILL.md``.
    SkillFormatError
        If a ``SKILL.md`` is not UTF-8 or has no valid frontmatter.
    SkillConflictError
        If two different files define the same skill name.
    """

    def __init__(self, roots: Iterable[str | Path]) -> None:
        self.roots = [Path(root) for root in roots]
        self._skills = self._load_skills()

    def _load_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for root in self.roots:
            if not root.exists():
                raise FileNotFoundError(f"Skill root {root} does not exist")
            if root.is_file() and not _is_skill_file(root):
                raise ValueError(
                    f"Skill root {root} is a file but not a SKILL.md; pass its folder"
                )
            paths = [root] if _is_skill_file(root) else sorted(root.rglob("SKILL.md"))
            if not paths:
                raise FileNotFoundError(f"Skill root {root} holds no SKILL.md file")
            for path in paths:
                try:
                    content = path.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError as error:
                    raise SkillFormatError(f"{path} is not valid UTF-8") from error
                frontmatter = _read_frontmatter(path, content)
                name = frontmatter["name"]
                existing = skills.get(name)
                if existing is not None and existing.path.resolve() != path.resolve():
                    raise SkillConflictError(
                        f"Skill '{name}' is defined twice: {existing.path} and {path}"
                    )
                skills[name] = Skill(
                    name=name,
                    description=frontmatter["description"],
                    path=path,
                    root_path=path.parent,
                    content=content,
                )
        return skills

    def list_skills(self) -> list[Skill]:
        """Return all skills sorted by name."""
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill:
        """Return a skill by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        Skill

        Raises
        ------
        SkillNotFoundError
        """
        if name not in self._skills:
            raise SkillNotFoundError(name)
        return self._skills[name]


class FilesystemSkills(BaseSkills):
    """List the skills in the requests of the named model nodes and offer ``load_skill``.

    The list is a ``SystemMessage`` inserted after the request's leading
    system messages; it is never stored.

    Parameters
    ----------
    roots : Iterable[str or Path] or SkillCatalog
        Roots as for ``SkillCatalog``, or a catalog already built.
    nodes : Collection[str]
        ``model_node`` names whose requests get the list, e.g. ``{"think"}``
        for ``build_react_agent``. Not checked against the graph: a misspelled
        name gets nothing.

    Raises
    ------
    TypeError
        If ``nodes`` is a single string.
    ValueError
        If ``nodes`` is empty, or a root is a file other than ``SKILL.md``.
    FileNotFoundError
        If a root does not exist or holds no ``SKILL.md``.
    SkillFormatError
        If a ``SKILL.md`` is not UTF-8 or has no valid frontmatter.
    SkillConflictError
        If two different files define the same skill name.
    """

    def __init__(
        self, roots: Iterable[str | Path] | SkillCatalog, *, nodes: Collection[str]
    ) -> None:
        if isinstance(nodes, str):
            raise TypeError(
                f"nodes must be a collection of node names, got the string {nodes!r}; "
                f"pass nodes={{{nodes!r}}}"
            )
        if not nodes:
            raise ValueError("nodes must name at least one node")
        self.catalog = roots if isinstance(roots, SkillCatalog) else SkillCatalog(roots)
        self.nodes = frozenset(nodes)
        self._tools = (self._build_load_skill(),)

    def list_skills(self) -> list[Skill]:
        """Return all skills sorted by name."""
        return self.catalog.list_skills()

    def get(self, name: str) -> Skill:
        """Return a skill by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        Skill

        Raises
        ------
        SkillNotFoundError
        """
        return self.catalog.get(name)

    def _build_load_skill(self) -> Tool:
        catalog = self.catalog

        @tool(name="load_skill", description="Load the full content of a named skill.")
        async def load_skill(
            name: str = Field(description="Skill name to load."),
        ) -> LoadedSkill:
            skill = catalog.get(name)
            return LoadedSkill(
                name=skill.name,
                description=skill.description,
                content=skill.content,
                path=str(skill.root_path),
                primary_file=str(skill.path),
            )

        return load_skill

    def tools(self) -> tuple[Tool, ...]:
        """Return the ``load_skill`` tool."""
        return self._tools

    def _format_catalog_prompt(self, skills: list[Skill]) -> str:
        lines = ["Available skills (call load_skill(name) to expand):"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def before_model(self, ctx: ModelMiddlewareContext) -> Replacement | None:
        """Add the skill catalog to the requests of the listed nodes.

        Returns
        -------
        Replacement or None
            The request with the catalog, or ``None`` for other nodes and an
            empty catalog.
        """
        skills = self.list_skills()
        if ctx.node_name not in self.nodes or not skills:
            return None
        messages = list(ctx.request.messages)
        position = 0
        while position < len(messages) and isinstance(
            messages[position], SystemMessage
        ):
            position += 1
        catalog = SystemMessage(content=self._format_catalog_prompt(skills))
        messages.insert(position, catalog)
        return ctx.replace(ctx.request.model_copy(update={"messages": messages}))


__all__ = [
    "BaseSkills",
    "FilesystemSkills",
    "LoadedSkill",
    "Skill",
    "SkillCatalog",
    "SkillConflictError",
    "SkillFormatError",
    "SkillNotFoundError",
]

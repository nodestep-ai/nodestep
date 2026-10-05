"""Built-in chat integrations: `ScriptedChat` for tests, and `OpenAIChat` in the `openai` extra.

Anthropic (Claude) and Ollama are planned. See
[Agents and tools](../../../concepts/agents.md#chat-models) for the table of
chat integrations, and [Test with ScriptedChat](../../../guides/testing.md).
"""

from importlib.util import find_spec
from typing import TYPE_CHECKING

from nodestep.chat.integrations.scripted import ScriptedChat

if TYPE_CHECKING:
    from nodestep.chat.integrations.openai import OpenAIChat as OpenAIChat
    from nodestep.chat.integrations.openai import OpenAISettings as OpenAISettings

__all__ = ["ScriptedChat"]


_LAZY_IMPORTS: dict[str, str] = {
    "OpenAIChat": "nodestep.chat.integrations.openai",
    "OpenAISettings": "nodestep.chat.integrations.openai",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is not None:
        import importlib

        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(
        f"module 'nodestep.chat.integrations' has no attribute {name!r}"
    )


def _openai_installed() -> bool:
    return all(find_spec(name) is not None for name in ("openai", "pydantic_settings"))


def __dir__() -> list[str]:
    """List the public names, and the OpenAI names when the ``openai`` extra is installed.

    The OpenAI names are not in ``__all__``, so a star import works without
    the extra, and ``dir`` lists them only when they can be imported, so
    ``inspect.getmembers`` and ``help`` work without it too.
    """
    return [*__all__, *(_LAZY_IMPORTS if _openai_installed() else ())]

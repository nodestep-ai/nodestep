"""Chat messages, requests and responses, and the `Chat` protocol for models.

A chat model is any object with `model`, `complete` and `stream`. See
[Agents and tools](../../concepts/agents.md#chat-models) for the chat
integrations, and [Write a chat integration](../../guides/chat-models.md).
"""

from nodestep.chat.base import Chat
from nodestep.chat.messages import (
    AIMessage,
    BaseMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    assemble_stream,
    parse_tool_arguments,
)

__all__ = [
    "AIMessage",
    "BaseMessage",
    "Chat",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "HumanMessage",
    "Message",
    "SystemMessage",
    "ToolCall",
    "ToolDefinition",
    "ToolMessage",
    "assemble_stream",
    "parse_tool_arguments",
]

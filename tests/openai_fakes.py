import json
from typing import Any

import httpx
import openai


def completion(
    content: str | None = None,
    tool_calls: list[tuple[str, str, Any]] | None = None,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "refusal": refusal,
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments),
                },
            }
            for call_id, name, arguments in tool_calls
        ]
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def chunk(
    delta: dict[str, Any] | None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "fake",
        "choices": []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


class FakeOpenAIServer:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode() or "{}"))
        item = self.script.pop(0)
        if isinstance(item, list):
            body = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in item)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )
        return httpx.Response(200, json=item)

    def client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="http://fake-openai.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )

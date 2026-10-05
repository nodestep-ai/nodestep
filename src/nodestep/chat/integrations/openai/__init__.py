"""The OpenAI chat integration: `OpenAIChat` for the OpenAI Chat Completions API and APIs compatible with it.

It has the `Chat` protocol, so it works wherever a chat model does; see
[Agents and tools](../../../concepts/agents.md#chat-models).

`OpenAIChat` is in the `openai` extra:

```bash
uv add "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep"
```

`import nodestep` works without the extra. `OpenAIChat` is exported lazily, and
using it without the extra raises `IntegrationNotInstalledError`, an `ImportError`
whose message shows the command above.

`OpenAIChat.from_env(model=...)` reads the variables below. Empty ones count as
unset, and keyword arguments override them.

| Variable | Setting | Default |
|---|---|---|
| `OPENAI_API_KEY` | `api_key` | required |
| `OPENAI_BASE_URL` | `base_url` | the OpenAI API |
| `OPENAI_ORG_ID` | `organization` | not sent |
| `OPENAI_PROJECT_ID` | `project` | not sent |
| `OPENAI_TIMEOUT` | `timeout` | 60 seconds |
| `OPENAI_MAX_RETRIES` | `max_retries` | 3 |

These are the variables in the repository's `.env.example`. The OpenAI SDK
handles the timeout and the retries; its own defaults are 600 seconds and 2
retries. Nothing loads a `.env` file, so pass it to the command:

```bash
cp .env.example .env
uv run --env-file .env python app.py
```

Without a key, `from_env` raises `ModelProviderError`:

```python
from nodestep import ModelProviderError, OpenAIChat

try:
    OpenAIChat.from_env(model="gpt-6-luna")
except ModelProviderError as error:
    print(error)
```

```text
OpenAIChat.from_env found no API key: OPENAI_API_KEY is not set or is empty; set it, or pass api_key=
```

In an agent, create the model where the graph is built, and keep `ScriptedChat`
in tests:

```diff
-chat = ScriptedChat([...])
+chat = OpenAIChat.from_env(model="gpt-6-luna")
```

`chat.requests` and `chat.responses` exist only on `ScriptedChat`, so drop the
lines that read them.

The OpenAI SDK reads some variables itself when it builds a client, with or
without `from_env`. nodestep does not patch the SDK.

- For a client you build, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID`
  and `OPENAI_PROJECT_ID` when you do not pass them.
- `OPENAI_CUSTOM_HEADERS` sets headers sent with every request.
- `OPENAI_LOG` sets the SDK's logging.
- `OPENAI_ADMIN_KEY` and `OPENAI_WEBHOOK_SECRET` are read too.
- httpx reads the proxy and CA certificate variables, such as `HTTPS_PROXY` and
  `SSL_CERT_FILE`.

How requests, responses and errors are mapped:

- SDK errors are raised as `ModelProviderError`, with the SDK's error as the
  cause.
- Retries happen inside the SDK and are not nodestep events.
- `output_schema=` uses OpenAI's strict mode, and the schema is rewritten for
  it. Every property becomes required, so the model writes fields that have a
  default too. Objects forbid extra properties, `"default": null` entries are
  removed, `$ref` entries with siblings are inlined, and the schema is sent
  under the name `final_output`.
- A message without content is sent as an empty string.
- A response with several choices (`n` > 1), or a tool call that is not a
  function call, raises `ModelProviderError`.
- `ChatResponse.model` is the model name the API reported; `ctx.model` in
  middleware is the model that was requested.
"""

from nodestep.chat.integrations.openai.chat import OpenAIChat
from nodestep.chat.integrations.openai.settings import OpenAISettings

__all__ = ["OpenAIChat", "OpenAISettings"]

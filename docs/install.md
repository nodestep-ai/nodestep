# Install

nodestep needs Python 3.12 or later, and you install it from GitHub because it is not on PyPI yet.

## Add it to a project

```bash
uv add "nodestep @ git+https://github.com/nodestep-ai/nodestep"
```

With pip: `pip install "nodestep @ git+https://github.com/nodestep-ai/nodestep"`. The requirements below work with `pip install` the same way.

## Extras

| Extra | Adds |
|---|---|
| `openai` | `OpenAIChat`, the [OpenAI chat integration](reference/chat/integrations/openai.md) |
| `sandbox` | the `nodestep sandbox` command, a local web page for running a graph (Starlette, Jinja2, uvicorn, Typer) |
| `all` | every extra above |

Name the extra in the Git URL:

```bash
uv add "nodestep[openai] @ git+https://github.com/nodestep-ai/nodestep"
```

The sandbox is a development tool, so add it as a dev dependency:

```bash
uv add --dev "nodestep[sandbox] @ git+https://github.com/nodestep-ai/nodestep"
```

Classes that need an extra are exported lazily, so `import nodestep` works without it. Using one without its extra, such as `OpenAIChat` without `openai`, raises `IntegrationNotInstalledError`, an `ImportError` whose message shows the install command.

## Pin a version

A Git URL without a reference follows the default branch. To stay on a tag or a commit, add it after a second `@`:

```bash
uv add "nodestep @ git+https://github.com/nodestep-ai/nodestep@<tag-or-commit>"
```

Either way, uv records the resolved commit in `uv.lock`, so `uv sync --locked` installs the same code until you run `uv lock --upgrade-package nodestep`.

## Run the sandbox without installing

```bash
uvx --from "nodestep[sandbox] @ git+https://github.com/nodestep-ai/nodestep" nodestep sandbox app.py:graph
```

`app.py:graph` is the file and the graph in it; the [Sandbox CLI](reference/sandbox-cli.md) page lists the other forms. `uvx` runs the command in a temporary environment with only nodestep and the sandbox extra. Name any other extra the graph needs, such as `openai`:

```bash
uvx --from "nodestep[openai,sandbox] @ git+https://github.com/nodestep-ai/nodestep" nodestep sandbox app.py:graph
```

Add other packages with `--with <package>`, or add the sandbox to your project as a dev dependency and run `uv run nodestep sandbox app.py:graph`.

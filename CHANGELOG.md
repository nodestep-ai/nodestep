# Changelog

## [0.1.0a1] - 2026-10-05

First alpha. Python 3.12 to 3.14.

### Added

- Graphs of sync or async nodes over a typed state, with conditional routes, fan-out with `Send` and a Mermaid drawing of the flow.
- Runs with `invoke` and `stream`, ready nodes in parallel, timeouts and a step limit.
- Threads, history and time travel, in memory or in files, and interrupts that pause a run for a person.
- Chat agents with tools, structured output and sub-agents. `OpenAIChat` in the `openai` extra, `ScriptedChat` for tests.
- Middleware for tool approvals, tool-call limits, summarization, to-do lists, skills and memory.
- Workspaces for file tools, on disk or in memory.
- `nodestep sandbox` in the `sandbox` extra: a local web page that runs a graph step by step.

[0.1.0a1]: https://github.com/nodestep-ai/nodestep/releases/tag/v0.1.0a1

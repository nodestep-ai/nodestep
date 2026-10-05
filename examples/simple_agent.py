import asyncio

from nodestep import ScriptedChat, build_react_agent, run_agent

chat = ScriptedChat(default="The capital of France is Paris.")

graph = build_react_agent(
    chat,
    tools=[],
    system_prompt="You are a helpful assistant.",
)


async def main() -> None:
    answer = await run_agent(graph, "What is the capital of France?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())

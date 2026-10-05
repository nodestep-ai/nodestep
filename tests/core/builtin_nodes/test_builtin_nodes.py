import pytest

from nodestep.core.node import get_node_metadata, node


@pytest.mark.asyncio
async def test_node_metadata_and_call() -> None:
    @node(name="route")
    async def router(state: dict) -> dict:
        return state

    assert get_node_metadata(router).name == "route"
    assert await router({}) == {}

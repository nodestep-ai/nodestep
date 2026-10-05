import asyncio


def pytest_configure() -> None:
    asyncio.set_event_loop(None)

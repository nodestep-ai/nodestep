import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from nodestep import END, START, BaseState, Command, Graph, Send, add, node

PRICES = {"north": 21.0, "south": 17.5, "west": 19.9}


class Offer(BaseModel):
    shop: str
    price: float


class PriceSearch(BaseState):
    product: str = "desk lamp"
    shops: list[str] = Field(default_factory=lambda: list(PRICES))
    shop: str = ""
    offers: Annotated[list[Offer], add] = Field(default_factory=list)
    best_shop: str = ""
    best_price: float = 0.0


@node
async def check_shop(state: PriceSearch) -> dict:
    await asyncio.sleep(0.01)
    return {"offers": [Offer(shop=state.shop, price=PRICES[state.shop])]}


@node(goto=[check_shop])
def fan_out(state: PriceSearch) -> Command:
    return Command(goto=[Send(check_shop, {"shop": shop}) for shop in state.shops])


@node
def pick_best(state: PriceSearch) -> dict:
    best = min(state.offers, key=lambda offer: offer.price)
    return {"best_shop": best.shop, "best_price": best.price}


graph = Graph(PriceSearch, name="price-search").flow(
    START >> fan_out,
    check_shop >> pick_best,
    pick_best >> END,
)


def main() -> None:
    result = graph.invoke({"product": "desk lamp"})
    for offer in sorted(result.state.offers, key=lambda offer: offer.shop):
        print(f"{offer.shop}: {offer.price}")
    state = result.state
    print(f"cheapest {state.product}: {state.best_shop} at {state.best_price}")


if __name__ == "__main__":
    main()

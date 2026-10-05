# Fan out with Send

A node that returns a `Command` with one `Send` per item runs another node once for each item, all in the same superstep.

## When you need it

Use a fan-out when the same work runs for every item of a list, such as a price check per shop or research per topic.

## Steps

1. Give the field that collects the results a merging reducer, such as `Annotated[list[Offer], add]`.
2. Add a field for the input of each task, here `shop: str = ""`. A `Send` payload may set only state fields.
3. Declare the fan-out node with `@node(goto=[check_shop])`, so `flow()` knows the target.
4. Return `Command(goto=[Send(check_shop, {"shop": shop}) for shop in state.shops])` from it.
5. Connect `check_shop >> pick_best`; `pick_best` runs once, after all the checks.
6. Run the graph.

## Complete example

`fan_out` sends `check_shop` to three shops, the `add` reducer collects the offers, and `pick_best` picks the cheapest:

```python
from typing import Annotated

from pydantic import BaseModel, Field

from nodestep import END, START, BaseState, Command, Graph, Send, add, node

PRICES = {"north": 21.0, "south": 17.5, "west": 19.9}


class Offer(BaseModel):
    shop: str
    price: float


class PriceSearch(BaseState):
    product: str = ""
    shops: list[str] = Field(default_factory=list)
    shop: str = ""
    offers: Annotated[list[Offer], add] = Field(default_factory=list)
    best_shop: str = ""
    best_price: float = 0.0


@node
def check_shop(state: PriceSearch) -> dict:
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

result = graph.invoke({"product": "desk lamp", "shops": list(PRICES)})
for offer in result.state.offers:
    print(f"{offer.shop}: {offer.price}")
state = result.state
print(f"cheapest {state.product}: {state.best_shop} at {state.best_price}")
```

```text
north: 21.0
south: 17.5
west: 19.9
cheapest desk lamp: south at 17.5
```

The three `check_shop` tasks run concurrently in superstep 1. Each one sees its own `shop`, and `pick_best` runs once in superstep 2.

## Good to know

- The payload is laid over the state for that task only: it skips the reducers and is not stored ([Command and Send](../concepts/nodes.md#command-and-send)).
- The payload's keys must be state fields with valid values ([Command and Send](../concepts/nodes.md#command-and-send)).
- On a field with the default `replace` reducer, the last `Send` task wins, so collect results with `add` ([Parallel writes](../concepts/execution.md#parallel-writes)).
- A node runs after any task routes to it, with no join barrier, so keep the paths to the collecting node the same length ([No join barrier](../concepts/execution.md#no-join-barrier)).

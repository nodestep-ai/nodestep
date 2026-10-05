from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NodestepModel(BaseModel):
    """Base pydantic model of nodestep's own models; unknown fields raise."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        revalidate_instances="never",
    )


class BaseState(BaseModel):
    """Base class for pydantic graph state schemas, with pydantic's default config."""


__all__ = ["BaseState", "NodestepModel"]

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GpuTypeModel(BaseModel):
    """Typed representation of an available GPU type."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    display_name: str = Field(alias="displayName", default="")
    memory_in_gb: int = Field(alias="memoryInGb", default=0)
    secure_cloud: bool = Field(alias="secureCloud", default=False)
    community_cloud: bool = Field(alias="communityCloud", default=False)
    secure_price: float | None = Field(alias="securePrice", default=None)
    community_price: float | None = Field(alias="communityPrice", default=None)
    lowest_price: dict | None = Field(alias="lowestPrice", default=None)

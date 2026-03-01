from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NetworkVolumeModel(BaseModel):
    """Typed representation of a network volume from the RunPod REST API."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str = ""
    size: int = 0
    data_center_id: str = Field(alias="dataCenterId", default="")

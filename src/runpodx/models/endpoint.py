from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EndpointModel(BaseModel):
    """Typed representation of a serverless endpoint from the RunPod REST API."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str = ""
    template_id: str | None = Field(alias="templateId", default=None)
    gpu_ids: str | None = Field(alias="gpuIds", default=None)
    workers_min: int = Field(alias="workersMin", default=0)
    workers_max: int = Field(alias="workersMax", default=0)
    idle_timeout: int = Field(alias="idleTimeout", default=5)
    flash_boot: bool = Field(alias="flashBoot", default=True)

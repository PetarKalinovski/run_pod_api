from __future__ import annotations

from typing import Annotated

from pydantic import Field

from runpodx.configs.base import BaseConfig


class EndpointCreateConfig(BaseConfig):
    """Configuration for creating a RunPod serverless endpoint."""

    name: Annotated[str, Field(description="Endpoint name.")]
    template_id: Annotated[str, Field(description="Template ID to use.")]
    gpu_ids: Annotated[str, Field(description="GPU type ID string.")]
    workers_min: Annotated[int, Field(ge=0, description="Minimum active workers.")] = 0
    workers_max: Annotated[int, Field(ge=1, description="Maximum active workers.")] = 3
    idle_timeout: Annotated[
        int, Field(ge=0, description="Seconds before idle workers scale down.")
    ] = 5
    flash_boot: Annotated[bool, Field(description="Enable flash boot for faster cold starts.")] = (
        True
    )
    scaling_type: Annotated[
        str | None, Field(description="Scaling strategy (e.g., 'QUEUE_DELAY').")
    ] = None

    def to_api_payload(self) -> dict:
        payload: dict = {
            "name": self.name,
            "templateId": self.template_id,
            "gpuIds": self.gpu_ids,
            "workersMin": self.workers_min,
            "workersMax": self.workers_max,
            "idleTimeout": self.idle_timeout,
            "flashBoot": self.flash_boot,
        }
        if self.scaling_type is not None:
            payload["scalingType"] = self.scaling_type
        return payload


class EndpointUpdateConfig(BaseConfig):
    """Configuration for updating an existing endpoint."""

    workers_min: Annotated[int | None, Field(ge=0, description="New min workers.")] = None
    workers_max: Annotated[int | None, Field(ge=1, description="New max workers.")] = None
    idle_timeout: Annotated[int | None, Field(ge=0, description="New idle timeout.")] = None

    def to_api_payload(self) -> dict:
        payload: dict = {}
        if self.workers_min is not None:
            payload["workersMin"] = self.workers_min
        if self.workers_max is not None:
            payload["workersMax"] = self.workers_max
        if self.idle_timeout is not None:
            payload["idleTimeout"] = self.idle_timeout
        return payload

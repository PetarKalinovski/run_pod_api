from __future__ import annotations

from typing import Annotated

from pydantic import Field

from runpodx.configs.base import BaseConfig


class NetworkVolumeCreateConfig(BaseConfig):
    """Configuration for creating a RunPod network volume."""

    name: Annotated[str, Field(description="Volume name.")]
    size_gb: Annotated[int, Field(ge=1, le=4000, description="Volume size in GB.")]
    data_center_id: Annotated[str, Field(description="Data center ID for the volume.")]

    def to_api_payload(self) -> dict:
        return {
            "name": self.name,
            "size": self.size_gb,
            "dataCenterId": self.data_center_id,
        }


class NetworkVolumeUpdateConfig(BaseConfig):
    """Configuration for updating an existing network volume."""

    name: Annotated[str | None, Field(description="New volume name.")] = None
    size_gb: Annotated[int | None, Field(ge=1, le=4000, description="New size in GB.")] = None

    def to_api_payload(self) -> dict:
        payload: dict = {}
        if self.name is not None:
            payload["name"] = self.name
        if self.size_gb is not None:
            payload["size"] = self.size_gb
        return payload

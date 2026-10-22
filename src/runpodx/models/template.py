from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TemplateModel(BaseModel):
    """Typed representation of a template from the RunPod REST API."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str = ""
    image_name: str = Field(alias="imageName", default="")
    is_serverless: bool = Field(alias="isServerless", default=False)
    container_disk_in_gb: int | None = Field(alias="containerDiskInGb", default=None)
    volume_in_gb: int | None = Field(alias="volumeInGb", default=None)
    volume_mount_path: str | None = Field(alias="volumeMountPath", default=None)
    ports: str | list[str] | None = None
    env: list[dict[str, Any]] | dict[str, Any] | None = None

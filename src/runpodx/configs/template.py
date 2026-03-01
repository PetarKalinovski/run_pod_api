from __future__ import annotations

from typing import Annotated

from pydantic import Field

from runpodx.configs.base import BaseConfig


class TemplateCreateConfig(BaseConfig):
    """Configuration for creating a RunPod template."""

    name: Annotated[str, Field(description="Template name.")]
    image: Annotated[str, Field(description="Container image tag.")]
    is_serverless: Annotated[
        bool, Field(description="Whether this template is for serverless endpoints.")
    ] = False
    container_disk_gb: Annotated[
        int, Field(ge=1, description="Container disk size in GB.")
    ] = 50
    volume_gb: Annotated[int, Field(ge=0, description="Persistent volume in GB.")] = 0
    volume_mount_path: Annotated[str, Field(description="Volume mount path.")] = "/workspace"
    ports: Annotated[str | None, Field(description="Exposed ports string.")] = None
    env: Annotated[dict[str, str], Field(description="Environment variables.")] = {}
    docker_start_cmd: Annotated[
        str | None, Field(description="Docker start command.")
    ] = None

    def to_api_payload(self) -> dict:
        payload: dict = {
            "name": self.name,
            "imageName": self.image,
            "isServerless": self.is_serverless,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": self.volume_mount_path,
            "env": self.env,
        }
        if self.ports is not None:
            payload["ports"] = self.ports
        if self.docker_start_cmd is not None:
            payload["dockerStartCmd"] = self.docker_start_cmd
        return payload


class TemplateUpdateConfig(BaseConfig):
    """Configuration for updating an existing template."""

    name: Annotated[str | None, Field(description="New template name.")] = None
    image: Annotated[str | None, Field(description="New container image.")] = None
    container_disk_gb: Annotated[
        int | None, Field(ge=1, description="New disk size in GB.")
    ] = None

    def to_api_payload(self) -> dict:
        payload: dict = {}
        if self.name is not None:
            payload["name"] = self.name
        if self.image is not None:
            payload["imageName"] = self.image
        if self.container_disk_gb is not None:
            payload["containerDiskInGb"] = self.container_disk_gb
        return payload

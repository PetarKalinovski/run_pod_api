from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GpuInfo(BaseModel):
    """GPU hardware info from the API response."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(alias="id", default="")
    count: int = Field(alias="gpuCount", default=1)


class RuntimeInfo(BaseModel):
    """Runtime details from the API response."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    uptime_in_seconds: int | None = Field(alias="uptimeInSeconds", default=None)
    ports: list[dict] | None = Field(alias="ports", default=None)
    gpu_count: int | None = Field(alias="gpuCount", default=None)


class PodModel(BaseModel):
    """Typed representation of a pod returned by the RunPod REST API.

    Uses field aliases to map from RunPod's camelCase JSON to snake_case Python.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str = ""
    desired_status: str = Field(alias="desiredStatus", default="")
    image_name: str = Field(alias="imageName", default="")
    machine_id: str | None = Field(alias="machineId", default=None)
    machine: dict | None = None

    # GPU info
    gpu_type_id: str | None = Field(alias="gpuTypeId", default=None)
    gpu_count: int = Field(alias="gpuCount", default=1)

    # Storage
    container_disk_in_gb: int | None = Field(alias="containerDiskInGb", default=None)
    volume_in_gb: int | None = Field(alias="volumeInGb", default=None)
    volume_mount_path: str | None = Field(alias="volumeMountPath", default=None)
    network_volume_id: str | None = Field(alias="networkVolumeId", default=None)

    # Networking
    public_ip: str | None = Field(alias="publicIp", default=None)
    ports: str | list | None = None
    port_mappings: dict[str, int] | None = Field(alias="portMappings", default=None)

    # Pricing
    cost_per_hr: float = Field(alias="costPerHr", default=0.0)
    cost_multiplier: float | None = Field(alias="costMultiplier", default=None)

    # Cloud info
    cloud_type: str | None = Field(alias="cloudType", default=None)
    compute_type: str | None = Field(alias="computeType", default=None)
    data_center_id: str | None = Field(alias="dataCenterId", default=None)

    # Template
    template_id: str | None = Field(alias="templateId", default=None)

    # Runtime
    runtime: RuntimeInfo | None = None

    # Env
    env: list[dict] | dict | None = None

    @property
    def is_running(self) -> bool:
        return self.desired_status == "RUNNING"

    @property
    def is_stopped(self) -> bool:
        return self.desired_status == "EXITED"

    @property
    def is_terminated(self) -> bool:
        return self.desired_status == "TERMINATED"

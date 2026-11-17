from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field


class GpuInfo(BaseModel):
    """A pod's GPU, as REST v2 reports it (``gpu: {id, count, vcpuCount, memory}``)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = ""
    count: int = Field(default=1, validation_alias=AliasChoices("count", "gpuCount"))
    vcpu_count: int | None = Field(alias="vcpuCount", default=None)
    memory: int | None = None


class RuntimeInfo(BaseModel):
    """A running pod's live details (v2 ``runtime``): its uptime and its port mappings,
    each ``{"private": 22, "public": 13607, "type": "tcp", "ip": "1.2.3.4"}``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    uptime: int | None = Field(default=None, validation_alias=AliasChoices("uptime", "uptimeInSeconds"))
    ports: list[dict] | None = None
    gpus: list[dict] | None = None


class PodModel(BaseModel):
    """Typed representation of a pod returned by the RunPod REST API v2.

    v2 renamed and nested much of v1's pod (``desiredStatus`` is ``status``, ``imageName``
    is ``image``, ``costPerHr`` is ``cost``, the GPU and the storage nested, the public IP
    and port mappings under ``runtime.ports``). v1's names stay readable as properties
    (``desired_status``, ``image_name``, ``gpu_type_id``, ``gpu_count``, ``cost_per_hr``,
    ``network_volume_id``, ``public_ip``, ``port_mappings``), and ``publicIp`` and
    ``portMappings`` are in ``model_dump(by_alias=True)`` as v1 had them, so callers that read
    those keep working.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str = ""
    status: str = Field(default="", validation_alias=AliasChoices("status", "desiredStatus", "desired_status"))
    image: str = Field(default="", validation_alias=AliasChoices("image", "imageName", "image_name"))
    disk: int | None = Field(default=None, validation_alias=AliasChoices("disk", "containerDiskInGb"))
    ports: list[str] | str | None = None
    env: dict | list | None = None
    gpu: GpuInfo | None = None
    cpu: dict | None = None
    mounts: dict | None = None
    cloud: str | None = Field(default=None, validation_alias=AliasChoices("cloud", "cloudType"))
    data_center_id: str | None = Field(alias="dataCenterId", default=None)
    cuda_version: str | None = Field(alias="cudaVersion", default=None)
    ssh: dict | None = None
    template: str | None = Field(default=None, validation_alias=AliasChoices("template", "templateId"))
    cost: float = Field(default=0.0, validation_alias=AliasChoices("cost", "costPerHr", "cost_per_hr"))
    locked: bool | None = None
    actions: list[str] | None = None
    runtime: RuntimeInfo | None = None
    created_at: str | None = Field(alias="createdAt", default=None)
    started_at: str | None = Field(alias="startedAt", default=None)

    # --- v1's names ---------------------------------------------------------------------
    @property
    def desired_status(self) -> str:
        return self.status

    @property
    def image_name(self) -> str:
        return self.image

    @property
    def gpu_type_id(self) -> str | None:
        if self.gpu and self.gpu.id:
            return self.gpu.id
        return (self.model_extra or {}).get("gpuTypeId")

    @property
    def gpu_count(self) -> int:
        return self.gpu.count if self.gpu else int((self.model_extra or {}).get("gpuCount", 1))

    @property
    def cost_per_hr(self) -> float:
        return self.cost

    @property
    def container_disk_in_gb(self) -> int | None:
        return self.disk

    @property
    def network_volume_id(self) -> str | None:
        net = (self.mounts or {}).get("network") or []
        if net:
            return net[0].get("volumeId")
        return (self.model_extra or {}).get("networkVolumeId")

    @computed_field(alias="publicIp")
    @property
    def public_ip(self) -> str | None:
        """The pod's public address: its published TCP ports' IP (v2 ``runtime.ports``),
        else the direct SSH endpoint's host."""
        for p in (self.runtime.ports if self.runtime and self.runtime.ports else []):
            if p.get("ip") and p.get("public") and str(p.get("type", "tcp")).lower() == "tcp":
                return p["ip"]
        direct = (self.ssh or {}).get("direct") or {}
        if direct.get("host"):
            return direct["host"]
        return (self.model_extra or {}).get("publicIp")

    @computed_field(alias="portMappings")
    @property
    def port_mappings(self) -> dict[str, int] | None:
        """Private port -> public port (``{"22": 13607}``), as v1 gave them."""
        out: dict[str, int] = {}
        for p in (self.runtime.ports if self.runtime and self.runtime.ports else []):
            if p.get("private") is not None and p.get("public"):
                out[str(p["private"])] = int(p["public"])
        direct = (self.ssh or {}).get("direct") or {}
        if "22" not in out and direct.get("port"):
            out["22"] = int(direct["port"])
        if not out:
            return (self.model_extra or {}).get("portMappings")
        return out

    @property
    def is_running(self) -> bool:
        return self.status == "RUNNING"

    @property
    def is_stopped(self) -> bool:
        return self.status == "EXITED"

    @property
    def is_terminated(self) -> bool:
        return self.status == "TERMINATED"

    @property
    def is_error(self) -> bool:
        return self.status == "ERROR"

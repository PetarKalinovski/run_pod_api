from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from runpodx.configs.base import BaseConfig


class GpuConfig(BaseConfig):
    """GPU hardware configuration for a pod."""

    count: Annotated[int, Field(ge=1, description="Number of GPUs to allocate.")] = 1
    type_ids: Annotated[
        list[str] | None,
        Field(
            description=(
                "Ordered list of acceptable GPU type IDs. "
                "Examples: 'NVIDIA GeForce RTX 4090', 'NVIDIA H100 80GB HBM3'."
            ),
        ),
    ] = None
    type_priority: Annotated[
        Literal["availability", "custom"],
        Field(description="How to prioritize GPU type selection."),
    ] = "availability"
    min_ram_per_gpu: Annotated[int, Field(ge=1, description="Minimum RAM in GB per GPU.")] = 8
    min_vcpu_per_gpu: Annotated[int, Field(ge=1, description="Minimum vCPUs per GPU.")] = 2
    allowed_cuda_versions: Annotated[
        list[str] | None,
        Field(description="Restrict to specific CUDA versions (e.g., '12.0', '12.4')."),
    ] = None


class StorageConfig(BaseConfig):
    """Disk and volume configuration for a pod."""

    container_disk_gb: Annotated[
        int, Field(ge=1, description="Ephemeral container disk in GB.")
    ] = 50
    volume_gb: Annotated[int, Field(ge=0, description="Persistent pod volume in GB.")] = 20
    volume_mount_path: Annotated[
        str, Field(description="Filesystem path where volumes are mounted.")
    ] = "/workspace"
    network_volume_id: Annotated[
        str | None,
        Field(description="Attach an existing network volume by its ID."),
    ] = None


class NetworkingConfig(BaseConfig):
    """Network and port configuration for a pod."""

    ports: Annotated[
        list[str],
        Field(description='Exposed ports, e.g. ["8888/http", "22/tcp"].'),
    ] = ["8888/http", "22/tcp"]
    data_center_ids: Annotated[
        list[str] | None,
        Field(description="Preferred data center IDs, e.g. ['US-TX-3', 'EU-RO-1']."),
    ] = None
    data_center_priority: Annotated[
        Literal["availability", "custom"],
        Field(description="How to prioritize data center selection."),
    ] = "availability"
    country_codes: Annotated[
        list[str] | None,
        Field(description="Restrict deployment to specific countries."),
    ] = None
    support_public_ip: Annotated[
        bool | None,
        Field(description="Require public IP (Community Cloud)."),
    ] = None


class PodCreateConfig(BaseConfig):
    """Full configuration for creating a RunPod GPU pod.

    Can be loaded from a TOML file:
        client.pods.create("pod.toml")

    TOML example:
        name = "my-training-pod"
        image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        cloud_type = "SECURE"

        [gpu]
        count = 2
        type_ids = ["NVIDIA GeForce RTX 4090"]

        [storage]
        container_disk_gb = 100
        volume_gb = 50

        [networking]
        ports = ["8888/http", "22/tcp"]
    """

    name: Annotated[str, Field(max_length=191, description="User-defined pod name.")] = "my pod"
    image: Annotated[str, Field(description="Container image tag to run.")]
    template_id: Annotated[
        str | None, Field(description="Template ID to use instead of manual config.")
    ] = None
    compute_type: Annotated[
        Literal["GPU", "CPU"], Field(description="Compute type: GPU or CPU.")
    ] = "GPU"
    cloud_type: Annotated[
        Literal["SECURE", "COMMUNITY"],
        Field(description="Cloud type: SECURE or COMMUNITY."),
    ] = "SECURE"

    gpu: Annotated[GpuConfig, Field(description="GPU hardware configuration.")] = GpuConfig()
    storage: Annotated[
        StorageConfig, Field(description="Disk and volume settings.")
    ] = StorageConfig()
    networking: Annotated[
        NetworkingConfig, Field(description="Network and port settings.")
    ] = NetworkingConfig()

    env: Annotated[
        dict[str, str], Field(description="Environment variables as key-value pairs.")
    ] = {}
    docker_entrypoint: Annotated[
        list[str], Field(description="Override container ENTRYPOINT.")
    ] = []
    docker_start_cmd: Annotated[
        list[str], Field(description="Override container CMD / start command.")
    ] = []

    interruptible: Annotated[
        bool, Field(description="Use spot/interruptible pricing.")
    ] = False
    locked: Annotated[
        bool, Field(description="Lock pod to prevent accidental stop/reset.")
    ] = False
    container_registry_auth_id: Annotated[
        str | None,
        Field(description="Container registry credentials ID for private images."),
    ] = None

    def to_api_payload(self) -> dict:
        """Convert to the RunPod REST API request body format."""
        payload: dict = {
            "name": self.name,
            "imageName": self.image,
            "computeType": self.compute_type,
            "cloudType": self.cloud_type,
            "gpuCount": self.gpu.count,
            "minRAMPerGPU": self.gpu.min_ram_per_gpu,
            "minVCPUPerGPU": self.gpu.min_vcpu_per_gpu,
            "containerDiskInGb": self.storage.container_disk_gb,
            "volumeInGb": self.storage.volume_gb,
            "volumeMountPath": self.storage.volume_mount_path,
            "ports": self.networking.ports,
            "interruptible": self.interruptible,
            "locked": self.locked,
            "env": self.env,
        }
        if self.template_id is not None:
            payload["templateId"] = self.template_id
        if self.gpu.type_ids is not None:
            payload["gpuTypeIds"] = self.gpu.type_ids
            payload["gpuTypePriority"] = self.gpu.type_priority
        if self.gpu.allowed_cuda_versions is not None:
            payload["allowedCudaVersions"] = self.gpu.allowed_cuda_versions
        if self.storage.network_volume_id is not None:
            payload["networkVolumeId"] = self.storage.network_volume_id
        if self.networking.data_center_ids is not None:
            payload["dataCenterIds"] = self.networking.data_center_ids
            payload["dataCenterPriority"] = self.networking.data_center_priority
        if self.networking.country_codes is not None:
            payload["countryCodes"] = self.networking.country_codes
        if self.networking.support_public_ip is not None:
            payload["supportPublicIp"] = self.networking.support_public_ip
        if self.docker_entrypoint:
            payload["dockerEntrypoint"] = self.docker_entrypoint
        if self.docker_start_cmd:
            payload["dockerStartCmd"] = self.docker_start_cmd
        if self.container_registry_auth_id is not None:
            payload["containerRegistryAuthId"] = self.container_registry_auth_id
        return payload


class PodUpdateConfig(BaseConfig):
    """Configuration for updating an existing pod."""

    image: Annotated[str | None, Field(description="New container image.")] = None
    container_disk_gb: Annotated[
        int | None, Field(ge=1, description="New container disk size in GB.")
    ] = None
    volume_gb: Annotated[
        int | None, Field(ge=0, description="New persistent volume size in GB.")
    ] = None
    gpu_count: Annotated[int | None, Field(ge=1, description="New GPU count.")] = None

    def to_api_payload(self) -> dict:
        """Convert to the RunPod REST API request body format."""
        payload: dict = {}
        if self.image is not None:
            payload["imageName"] = self.image
        if self.container_disk_gb is not None:
            payload["containerDiskInGb"] = self.container_disk_gb
        if self.volume_gb is not None:
            payload["volumeInGb"] = self.volume_gb
        if self.gpu_count is not None:
            payload["gpuCount"] = self.gpu_count
        return payload

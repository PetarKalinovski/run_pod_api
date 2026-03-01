from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from runpodx.configs.loader import resolve_config
from runpodx.configs.pod import PodCreateConfig, PodUpdateConfig
from runpodx.http.errors import RunPodError, TimeoutError
from runpodx.models.pod import PodModel

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


class Pod:
    """A live pod resource with action methods.

    Provides the resource-oriented API:
        pod = client.pods.get("pod_abc123")
        pod.stop()
        pod.start()
        pod.wait_until_ready()
        pod.delete()
    """

    def __init__(self, data: PodModel, transport: HttpTransport) -> None:
        self._data = data
        self._transport = transport

    def __getattr__(self, name: str):
        return getattr(self._data, name)

    @property
    def data(self) -> PodModel:
        """Access the full typed pod data model."""
        return self._data

    def refresh(self) -> Pod:
        """Fetch the latest pod state from the API."""
        raw = self._transport.request("GET", f"/pods/{self._data.id}")
        self._data = PodModel.model_validate(raw)
        return self

    def start(self) -> Pod:
        """Start a stopped pod."""
        self._transport.request("POST", f"/pods/{self._data.id}/start")
        return self.refresh()

    def stop(self) -> Pod:
        """Stop a running pod (preserves data, stops billing)."""
        self._transport.request("POST", f"/pods/{self._data.id}/stop")
        return self.refresh()

    def restart(self) -> Pod:
        """Restart the pod."""
        self._transport.request("POST", f"/pods/{self._data.id}/restart")
        return self.refresh()

    def update(self, config: str | Path | dict | PodUpdateConfig) -> Pod:
        """Update pod configuration (may trigger a reset)."""
        resolved = resolve_config(PodUpdateConfig, config)
        payload = resolved.to_api_payload()
        raw = self._transport.request("PATCH", f"/pods/{self._data.id}", json=payload)
        self._data = PodModel.model_validate(raw)
        return self

    def delete(self) -> None:
        """Permanently delete the pod."""
        self._transport.request("DELETE", f"/pods/{self._data.id}")

    def wait_until_ready(self, timeout: float = 300, poll_interval: float = 5) -> Pod:
        """Poll until the pod reaches RUNNING status with a public IP.

        Args:
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between status checks.

        Raises:
            TimeoutError: If the pod does not become ready within the timeout.
            RunPodError: If the pod is terminated while waiting.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.refresh()
            if self._data.is_running and self._data.public_ip:
                return self
            if self._data.is_terminated:
                raise RunPodError(f"Pod {self._data.id} was terminated while waiting.")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Pod {self._data.id} did not become ready within {timeout}s. "
            f"Current status: {self._data.desired_status}"
        )

    def __repr__(self) -> str:
        return (
            f"Pod(id={self._data.id!r}, name={self._data.name!r}, "
            f"status={self._data.desired_status!r})"
        )


class PodManager:
    """Resource manager for pods. Accessed as client.pods.

    Usage:
        pods = client.pods.list()
        pod = client.pods.create("training.toml")
        pod = client.pods.get("pod_abc123")
    """

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        gpu_type_id: str | None = None,
        desired_status: str | None = None,
        name: str | None = None,
    ) -> list[Pod]:
        """List all pods, with optional filters."""
        params: dict = {}
        if gpu_type_id:
            params["gpuTypeId"] = gpu_type_id
        if desired_status:
            params["desiredStatus"] = desired_status
        if name:
            params["name"] = name
        raw_list = self._transport.request("GET", "/pods", params=params)
        return [Pod(PodModel.model_validate(item), self._transport) for item in raw_list]

    def get(self, pod_id: str) -> Pod:
        """Get a single pod by ID."""
        raw = self._transport.request("GET", f"/pods/{pod_id}")
        return Pod(PodModel.model_validate(raw), self._transport)

    def create(self, config: str | Path | dict | PodCreateConfig) -> Pod:
        """Create a new pod from a TOML file, dict, or config object.

        Examples:
            pod = client.pods.create("pod.toml")
            pod = client.pods.create({"image": "pytorch/pytorch:2.1.0", ...})
            pod = client.pods.create(PodCreateConfig(image="pytorch/pytorch:2.1.0"))
        """
        resolved = resolve_config(PodCreateConfig, config)
        payload = resolved.to_api_payload()
        raw = self._transport.request("POST", "/pods", json=payload)
        return Pod(PodModel.model_validate(raw), self._transport)

    def delete(self, pod_id: str) -> None:
        """Delete a pod by ID."""
        self._transport.request("DELETE", f"/pods/{pod_id}")

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from runpodx.configs.loader import resolve_config
from runpodx.configs.pod import PodCreateConfig, PodUpdateConfig
from runpodx.http.errors import AuthenticationError, RunPodError, TimeoutError
from runpodx.http.pages import collect
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

    def _action(self, action: str) -> Pod:
        # (REST v2: one endpoint for every state change, {"action": start|stop|restart|terminate})
        self._transport.request("POST", f"/pods/{self._data.id}/action", json={"action": action})
        return self.refresh()

    def start(self) -> Pod:
        """Start a stopped pod."""
        return self._action("start")

    def stop(self) -> Pod:
        """Stop a running pod (preserves data, stops billing)."""
        return self._action("stop")

    def restart(self) -> Pod:
        """Restart the pod."""
        return self._action("restart")

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
            if self._data.is_error:
                raise RunPodError(f"Pod {self._data.id} went into ERROR while waiting.")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Pod {self._data.id} did not become ready within {timeout}s. "
            f"Current status: {self._data.status}"
        )

    def __repr__(self) -> str:
        return (
            f"Pod(id={self._data.id!r}, name={self._data.name!r}, "
            f"status={self._data.status!r})"
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
        """List all pods (every page), with optional filters (applied here)."""
        pods = [PodModel.model_validate(item) for item in collect(self._transport, "/pods", "pods")]
        if gpu_type_id:
            pods = [m for m in pods if m.gpu_type_id == gpu_type_id]
        if desired_status:
            pods = [m for m in pods if m.status == desired_status]
        if name:
            pods = [m for m in pods if m.name == name]
        return [Pod(m, self._transport) for m in pods]

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

        REST v2 takes one GPU type a pod; with several ``gpu.type_ids`` each is tried in
        order until one is created (v1 took the list and chose itself). The last failure is
        raised if none can be had.
        """
        resolved = resolve_config(PodCreateConfig, config)
        types = list(resolved.gpu.type_ids or []) or [None]
        last: Exception | None = None
        for type_id in types:
            payload = resolved.to_api_payload(gpu_type_id=type_id)
            try:
                raw = self._transport.request("POST", "/pods", json=payload)
            except AuthenticationError:
                raise
            except RunPodError as e:
                last = e
                continue
            return Pod(PodModel.model_validate(raw), self._transport)
        raise last  # type: ignore[misc]

    def delete(self, pod_id: str) -> None:
        """Delete a pod by ID."""
        self._transport.request("DELETE", f"/pods/{pod_id}")

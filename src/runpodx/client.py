from __future__ import annotations

from runpodx.http.transport import HttpTransport, TransportConfig
from runpodx.resources.billing import BillingManager
from runpodx.resources.endpoints import EndpointManager
from runpodx.resources.network_volumes import NetworkVolumeManager
from runpodx.resources.pods import PodManager
from runpodx.resources.templates import TemplateManager


class RunPodClient:
    """Top-level client for the RunPod REST API.

    Usage:
        from runpodx import RunPodClient

        client = RunPodClient()                          # reads RUNPOD_API_KEY from env
        client = RunPodClient(api_key="rp_xxxxxx")       # explicit key

        pods = client.pods.list()
        pod = client.pods.create("pod.toml")
        pod.stop()
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        config = TransportConfig(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
            timeout=timeout,
            max_retries=max_retries,
        )
        self._transport = HttpTransport(config)

        self.pods = PodManager(self._transport)
        self.templates = TemplateManager(self._transport)
        self.network_volumes = NetworkVolumeManager(self._transport)
        self.endpoints = EndpointManager(self._transport)
        self.billing = BillingManager(self._transport)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

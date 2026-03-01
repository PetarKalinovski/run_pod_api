from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from runpodx.configs.loader import resolve_config
from runpodx.configs.network_volume import NetworkVolumeCreateConfig, NetworkVolumeUpdateConfig
from runpodx.models.network_volume import NetworkVolumeModel

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


class NetworkVolumeManager:
    """Resource manager for network volumes. Accessed as client.network_volumes."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> list[NetworkVolumeModel]:
        """List all network volumes."""
        raw = self._transport.request("GET", "/networkvolumes")
        return [NetworkVolumeModel.model_validate(item) for item in raw]

    def get(self, volume_id: str) -> NetworkVolumeModel:
        """Get a network volume by ID."""
        raw = self._transport.request("GET", f"/networkvolumes/{volume_id}")
        return NetworkVolumeModel.model_validate(raw)

    def create(
        self, config: str | Path | dict | NetworkVolumeCreateConfig
    ) -> NetworkVolumeModel:
        """Create a new network volume."""
        resolved = resolve_config(NetworkVolumeCreateConfig, config)
        raw = self._transport.request("POST", "/networkvolumes", json=resolved.to_api_payload())
        return NetworkVolumeModel.model_validate(raw)

    def update(
        self, volume_id: str, config: str | Path | dict | NetworkVolumeUpdateConfig
    ) -> NetworkVolumeModel:
        """Update an existing network volume."""
        resolved = resolve_config(NetworkVolumeUpdateConfig, config)
        raw = self._transport.request(
            "PATCH", f"/networkvolumes/{volume_id}", json=resolved.to_api_payload()
        )
        return NetworkVolumeModel.model_validate(raw)

    def delete(self, volume_id: str) -> None:
        """Delete a network volume."""
        self._transport.request("DELETE", f"/networkvolumes/{volume_id}")
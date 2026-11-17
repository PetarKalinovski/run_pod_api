from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from runpodx.configs.endpoint import EndpointCreateConfig, EndpointUpdateConfig
from runpodx.configs.loader import resolve_config
from runpodx.models.endpoint import EndpointModel

from runpodx.http.pages import collect

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


class EndpointManager:
    """Resource manager for serverless endpoints. Accessed as client.endpoints."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> list[EndpointModel]:
        """List all serverless endpoints."""
        # (REST v2: serverless endpoints at /v2/serverless, the list wrapped as {"endpoints": [...]})
        return [EndpointModel.model_validate(item) for item in collect(self._transport, "/serverless", "endpoints")]

    def create(self, config: str | Path | dict | EndpointCreateConfig) -> EndpointModel:
        """Create a new serverless endpoint."""
        resolved = resolve_config(EndpointCreateConfig, config)
        raw = self._transport.request("POST", "/serverless", json=resolved.to_api_payload())
        return EndpointModel.model_validate(raw)

    def update(
        self, endpoint_id: str, config: str | Path | dict | EndpointUpdateConfig
    ) -> EndpointModel:
        """Update an existing endpoint."""
        resolved = resolve_config(EndpointUpdateConfig, config)
        raw = self._transport.request(
            "PATCH", f"/serverless/{endpoint_id}", json=resolved.to_api_payload()
        )
        return EndpointModel.model_validate(raw)

    def delete(self, endpoint_id: str) -> None:
        """Delete a serverless endpoint."""
        self._transport.request("DELETE", f"/serverless/{endpoint_id}")

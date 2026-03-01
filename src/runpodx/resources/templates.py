from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from runpodx.configs.loader import resolve_config
from runpodx.configs.template import TemplateCreateConfig, TemplateUpdateConfig
from runpodx.models.template import TemplateModel

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


class TemplateManager:
    """Resource manager for templates. Accessed as client.templates."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        *,
        include_public: bool = False,
        include_runpod: bool = False,
    ) -> list[TemplateModel]:
        """List templates."""
        params = {
            "includePublicTemplates": include_public,
            "includeRunpodTemplates": include_runpod,
        }
        raw = self._transport.request("GET", "/templates", params=params)
        return [TemplateModel.model_validate(item) for item in raw]

    def create(self, config: str | Path | dict | TemplateCreateConfig) -> TemplateModel:
        """Create a new template."""
        resolved = resolve_config(TemplateCreateConfig, config)
        raw = self._transport.request("POST", "/templates", json=resolved.to_api_payload())
        return TemplateModel.model_validate(raw)

    def update(
        self, template_id: str, config: str | Path | dict | TemplateUpdateConfig
    ) -> TemplateModel:
        """Update an existing template."""
        resolved = resolve_config(TemplateUpdateConfig, config)
        raw = self._transport.request(
            "PATCH", f"/templates/{template_id}", json=resolved.to_api_payload()
        )
        return TemplateModel.model_validate(raw)

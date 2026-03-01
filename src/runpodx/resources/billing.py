from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from runpodx.models.billing import BillingRecord

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


class BillingManager:
    """Resource manager for billing data. Accessed as client.billing."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def pods(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        bucket_size: str = "day",
        pod_id: str | None = None,
        gpu_type_id: str | None = None,
        grouping: str = "gpuTypeId",
    ) -> list[BillingRecord]:
        """Get pod billing history."""
        params: dict = {"bucketSize": bucket_size, "grouping": grouping}
        if start_time:
            params["startTime"] = start_time.isoformat()
        if end_time:
            params["endTime"] = end_time.isoformat()
        if pod_id:
            params["podId"] = pod_id
        if gpu_type_id:
            params["gpuTypeId"] = gpu_type_id
        raw = self._transport.request("GET", "/billing/pods", params=params)
        return [BillingRecord.model_validate(item) for item in raw]

    def endpoints(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        bucket_size: str = "day",
        endpoint_id: str | None = None,
        grouping: str = "endpointId",
    ) -> list[BillingRecord]:
        """Get serverless endpoint billing history."""
        params: dict = {"bucketSize": bucket_size, "grouping": grouping}
        if start_time:
            params["startTime"] = start_time.isoformat()
        if end_time:
            params["endTime"] = end_time.isoformat()
        if endpoint_id:
            params["endpointId"] = endpoint_id
        raw = self._transport.request("GET", "/billing/endpoints", params=params)
        return [BillingRecord.model_validate(item) for item in raw]

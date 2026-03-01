from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BillingRecord(BaseModel):
    """A single billing record from the RunPod billing API."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    amount: float = 0.0
    discount_amount: float = Field(alias="discountAmount", default=0.0)
    pod_id: str | None = Field(alias="podId", default=None)
    gpu_type_id: str | None = Field(alias="gpuTypeId", default=None)
    start_time: str | None = Field(alias="startTime", default=None)
    end_time: str | None = Field(alias="endTime", default=None)
    type: str | None = None

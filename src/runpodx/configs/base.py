from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class BaseConfig(BaseModel):
    """Base for all runpodx configuration models.

    Uses Pydantic BaseModel for validation, serialization, and schema generation.
    Forbids extra fields and converts the string "None" in TOML to Python None.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("*", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "None":
            return None
        return v
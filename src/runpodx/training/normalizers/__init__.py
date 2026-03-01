from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

_NORMALIZER_REGISTRY: dict[tuple[str, str], type[BaseNormalizer]] = {}


def register_normalizer(method: str, format_name: str):
    """Decorator to register a normalizer for a (method, format) pair."""

    def decorator(cls: type[BaseNormalizer]):
        _NORMALIZER_REGISTRY[(method, format_name)] = cls
        return cls

    return decorator


def get_normalizer(method: str, format_name: str, **kwargs) -> BaseNormalizer:
    """Look up and instantiate the normalizer for the given method and format."""
    key = (method, format_name)
    if key not in _NORMALIZER_REGISTRY:
        available = [f"{m}:{f}" for m, f in _NORMALIZER_REGISTRY]
        raise ValueError(
            f"No normalizer for method={method!r}, format={format_name!r}. "
            f"Available: {available}"
        )
    return _NORMALIZER_REGISTRY[key](**kwargs)


class BaseNormalizer(ABC):
    """Base class for dataset normalizers."""

    @abstractmethod
    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert a list of raw records to canonical format."""
        ...

    @abstractmethod
    def validate_record(self, record: dict[str, Any]) -> bool:
        """Check whether a single record has the required fields."""
        ...


# Import concrete normalizers so decorators fire
from runpodx.training.normalizers import sft as _sft  # noqa: E402, F401

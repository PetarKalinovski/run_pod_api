from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from runpodx.training.configs import RunConfig

_METHOD_REGISTRY: dict[str, type[BaseMethod]] = {}


def register_method(name: str):
    """Decorator to register a training method."""

    def decorator(cls: type[BaseMethod]):
        _METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def get_method(name: str) -> BaseMethod:
    """Look up and instantiate the method for the given name."""
    if name not in _METHOD_REGISTRY:
        raise ValueError(
            f"Unknown training method: {name!r}. Available: {list(_METHOD_REGISTRY.keys())}"
        )
    return _METHOD_REGISTRY[name]()


class BaseMethod(ABC):
    """Base class for training methods (SFT, GRPO, etc.)."""

    @abstractmethod
    def script_path(self, run_config: RunConfig | None = None) -> Path:
        """Path to the training script to upload."""
        ...

    @abstractmethod
    def requirements_path(self, run_config: RunConfig | None = None) -> Path:
        """Path to the requirements.txt for this method."""
        ...

    @abstractmethod
    def generate_remote_config(self, run_config: RunConfig) -> dict:
        """Generate the JSON config that the remote script reads."""
        ...

    @abstractmethod
    def canonical_format(self) -> str:
        """Method name for normalizer lookup (e.g., 'sft')."""
        ...


# Import concrete methods so decorators fire
from runpodx.training.methods import sft as _sft  # noqa: E402, F401

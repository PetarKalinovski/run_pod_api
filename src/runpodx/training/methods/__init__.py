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
    def script_path(self, run_config: RunConfig | None = None) -> Path | None:
        """Path to the main script to upload, if this method has one."""
        ...

    def remote_script_name(self, run_config: RunConfig | None = None) -> str:
        """Filename used for the main script on the pod."""
        return "train.py"

    def requirements_path(self, run_config: RunConfig | None = None) -> Path | None:
        """Path to an optional requirements.txt for this method."""
        return None

    def support_paths(self, run_config: RunConfig | None = None) -> list[Path]:
        """Additional local files to upload next to the main script."""
        script_path = self.script_path(run_config)
        if script_path is None:
            return []
        return [
            support_file
            for support_file in script_path.parent.glob("*.py")
            if support_file != script_path
        ]

    def preinstall_commands(self, run_config: RunConfig | None = None) -> list[str]:
        """Commands to run before requirements installation."""
        return []

    def should_upload_config(self, run_config: RunConfig | None = None) -> bool:
        """Whether config.json should be uploaded for this method."""
        return True

    def remote_command(self, remote_work_dir: str, run_config: RunConfig) -> str:
        """Command to execute on the pod."""
        return (
            f"python {self.remote_script_name(run_config)} "
            f"--config {remote_work_dir}/config.json"
        )

    @abstractmethod
    def generate_remote_config(self, run_config: RunConfig) -> dict:
        """Generate the JSON config that the remote script reads."""
        ...

    @abstractmethod
    def canonical_format(self) -> str:
        """Method name for normalizer lookup (e.g., 'sft')."""
        ...


# Import concrete methods so decorators fire
from runpodx.training.methods import grpo as _grpo  # noqa: E402, F401
from runpodx.training.methods import script as _script  # noqa: E402, F401
from runpodx.training.methods import sft as _sft  # noqa: E402, F401

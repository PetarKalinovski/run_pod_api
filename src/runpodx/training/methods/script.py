from __future__ import annotations

from pathlib import Path

from runpodx.training.configs import RunConfig
from runpodx.training.methods import BaseMethod, register_method


@register_method("script")
class ScriptMethod(BaseMethod):
    """Generic script/command runner method."""

    def script_path(self, run_config: RunConfig | None = None) -> Path | None:
        if run_config is None or not run_config.script.entrypoint:
            return None
        return Path(run_config.script.entrypoint)

    def remote_script_name(self, run_config: RunConfig | None = None) -> str:
        script_path = self.script_path(run_config)
        if script_path is None:
            return "run.py"
        return script_path.name

    def requirements_path(self, run_config: RunConfig | None = None) -> Path | None:
        if run_config is None or not run_config.script.requirements:
            return None
        return Path(run_config.script.requirements)

    def support_paths(self, run_config: RunConfig | None = None) -> list[Path]:
        if run_config is None:
            return []
        return [Path(path) for path in run_config.script.uploads]

    def preinstall_commands(self, run_config: RunConfig | None = None) -> list[str]:
        if run_config is None:
            return []
        return list(run_config.script.preinstall_commands)

    def generate_remote_config(self, run_config: RunConfig) -> dict:
        return dict(run_config.script.config)

    def should_upload_config(self, run_config: RunConfig | None = None) -> bool:
        if run_config is None:
            return False
        return bool(run_config.script.config)

    def remote_command(self, remote_work_dir: str, run_config: RunConfig) -> str:
        config_path = f"{remote_work_dir}/config.json"
        script_path = f"{remote_work_dir}/{self.remote_script_name(run_config)}"
        if run_config.script.command:
            return run_config.script.command.format(
                remote_work_dir=remote_work_dir,
                script_path=script_path,
                config_path=config_path,
            )
        return f"python {script_path} --config {config_path}"

    def canonical_format(self) -> str:
        return "script"

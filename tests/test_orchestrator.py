from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from runpodx.models.pod import PodModel
from runpodx.resources.pods import Pod
from runpodx.training.configs import (
    DatasetConfig,
    ModelConfig,
    MonitoringConfig,
    PodSpecConfig,
    RunConfig,
    RunMetaConfig,
)
from runpodx.training.methods import get_method
from runpodx.training.orchestrator import TrainingOrchestrator


@pytest.fixture
def mock_pod_manager(sample_pod_data):
    manager = MagicMock()
    pod_model = PodModel.model_validate(sample_pod_data)
    mock_pod = MagicMock(spec=Pod)
    mock_pod.id = "pod_abc123"
    mock_pod.data = pod_model
    mock_pod.wait_until_ready.return_value = mock_pod
    manager.get.return_value = mock_pod
    manager.create.return_value = mock_pod
    return manager


@pytest.fixture
def sample_run_config():
    return RunConfig(
        run=RunMetaConfig(method="sft", pod_id="pod_abc123"),
        model=ModelConfig(name="Qwen/Qwen3-0.6B"),
        dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
        monitoring=MonitoringConfig(wandb_project="test"),
    )


@pytest.fixture
def sample_run_config_new_pod():
    return RunConfig(
        run=RunMetaConfig(method="sft"),
        pod=PodSpecConfig(),
        model=ModelConfig(name="Qwen/Qwen3-0.6B"),
        dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
    )


class TestSFTMethod:
    def test_generate_remote_config(self, sample_run_config):
        method = get_method("sft")
        remote_config = method.generate_remote_config(sample_run_config)
        assert remote_config["model_name"] == "Qwen/Qwen3-0.6B"
        assert remote_config["adapter"] == "lora"
        assert remote_config["lora_r"] == 16
        assert remote_config["num_train_epochs"] == 3
        assert remote_config["learning_rate"] == 2e-5
        assert remote_config["hf_dataset"] == "tatsu-lab/alpaca"
        assert remote_config["wandb_project"] == "test"

    def test_generate_remote_config_with_validation(self):
        config = RunConfig(
            run=RunMetaConfig(method="sft", pod_id="pod_abc123"),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            dataset=DatasetConfig(
                source="huggingface",
                path="tatsu-lab/alpaca",
                validation_ratio=0.1,
                validation_seed=7,
            ),
        )
        method = get_method("sft")
        remote_config = method.generate_remote_config(config)

        assert remote_config["validation_ratio"] == 0.1
        assert remote_config["validation_seed"] == 7

    def test_script_path_exists(self):
        method = get_method("sft")
        assert method.script_path().exists()
        assert method.requirements_path().exists()

    def test_trinity_uses_dedicated_script_and_requirements(self):
        config = RunConfig(
            run=RunMetaConfig(method="sft", pod_id="pod_abc123"),
            model=ModelConfig(name="arcee-ai/Trinity-Nano-Base", adapter="full"),
            dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
        )
        method = get_method("sft")

        assert method.script_path(config).name == "sft_train_trinity.py"
        assert method.requirements_path(config).name == "requirements_trinity.txt"

    def test_canonical_format(self):
        method = get_method("sft")
        assert method.canonical_format() == "sft"


class TestTrainingOrchestrator:
    def test_dry_run(self, mock_pod_manager, sample_run_config):
        orchestrator = TrainingOrchestrator(mock_pod_manager)
        # dry_run should not raise, should not call any pod methods
        orchestrator.run(sample_run_config, dry_run=True)
        mock_pod_manager.get.assert_not_called()
        mock_pod_manager.create.assert_not_called()

    def test_dry_run_new_pod(self, mock_pod_manager, sample_run_config_new_pod):
        orchestrator = TrainingOrchestrator(mock_pod_manager)
        orchestrator.run(sample_run_config_new_pod, dry_run=True)
        mock_pod_manager.create.assert_not_called()

    def test_pod_id_override(self, mock_pod_manager, sample_run_config):
        orchestrator = TrainingOrchestrator(mock_pod_manager)
        orchestrator.run(sample_run_config, pod_id_override="pod_xyz", dry_run=True)
        # The config should have been updated
        # (dry_run so no actual API calls)

    @patch("runpodx.training.orchestrator.SSHExecutor")
    @patch("runpodx.training.orchestrator.SSHConnection")
    def test_full_pipeline_with_existing_pod(
        self, mock_ssh_conn_cls, mock_executor_cls, mock_pod_manager, sample_run_config
    ):
        mock_executor = MagicMock()
        mock_executor.test_connection.return_value = True
        mock_executor.run.return_value = MagicMock(returncode=0)
        mock_executor.upload.return_value = MagicMock(returncode=0)
        mock_executor_cls.return_value = mock_executor

        orchestrator = TrainingOrchestrator(mock_pod_manager)

        # Mock the streaming process for training
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0
        # The first calls to run() with stream=False return CompletedProcess
        # The last call with stream=True returns the mock_proc
        mock_executor.run.side_effect = [
            MagicMock(returncode=0),  # mkdir
            MagicMock(returncode=0),  # pip install
            mock_proc,                # training
        ]

        orchestrator.run(sample_run_config)

        mock_pod_manager.get.assert_called_once_with("pod_abc123")
        assert mock_executor.upload.call_count >= 2  # script + requirements + config


class TestMethodRegistry:
    def test_get_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_method("nonexistent")

    def test_get_sft(self):
        method = get_method("sft")
        assert method.canonical_format() == "sft"

from __future__ import annotations

import os
import tempfile

import pytest

from runpodx.configs.loader import resolve_config
from runpodx.training.configs import (
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    PodSpecConfig,
    RunConfig,
    RunMetaConfig,
    ScriptConfig,
    TrainingConfig,
)


class TestRunMetaConfig:
    def test_valid_method(self):
        config = RunMetaConfig(method="sft")
        assert config.method == "sft"

    def test_hub_cleanup_defaults_to_disabled(self):
        config = RunMetaConfig(method="sft")
        assert config.terminate_after_hub_push is False

    def test_invalid_method(self):
        with pytest.raises(Exception):
            RunMetaConfig(method="invalid")

    def test_pod_id_optional(self):
        config = RunMetaConfig(method="sft")
        assert config.pod_id is None


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig(name="Qwen/Qwen3-0.6B")
        assert config.adapter == "lora"
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_full_finetune(self):
        config = ModelConfig(name="Qwen/Qwen3-0.6B", adapter="full")
        assert config.adapter == "full"


class TestDatasetConfig:
    def test_huggingface_source(self):
        config = DatasetConfig(source="huggingface", path="tatsu-lab/alpaca")
        assert config.source == "huggingface"
        assert config.format == "alpaca"

    def test_local_source(self):
        config = DatasetConfig(source="local", path="data/train.jsonl")
        assert config.source == "local"

    def test_custom_requires_mapping(self):
        with pytest.raises(Exception, match="field_mapping"):
            DatasetConfig(source="local", path="data.jsonl", format="custom")

    def test_custom_with_mapping(self):
        config = DatasetConfig(
            source="local",
            path="data.jsonl",
            format="custom",
            field_mapping={"instruction": "prompt", "output": "response"},
        )
        assert config.field_mapping["instruction"] == "prompt"

    def test_validation_split_defaults(self):
        config = DatasetConfig(source="huggingface", path="tatsu-lab/alpaca")
        assert config.validation_ratio is None
        assert config.validation_seed == 42

    def test_validation_ratio_bounds(self):
        with pytest.raises(Exception):
            DatasetConfig(
                source="huggingface",
                path="tatsu-lab/alpaca",
                validation_ratio=1.0,
            )


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 3
        assert config.lr == 2e-5
        assert config.bf16 is True
        assert config.append_eos_token is True
        assert config.sanitize_ascii_outputs is False
        assert config.generation_eval_enabled is False
        assert config.convert_to_prompt_completion is False
        assert config.grpo_num_generations == 4
        assert config.grpo_max_completion_length == 192
        assert config.grpo_soft_overlong_cache == 32
        assert config.grpo_beta == 0.0
        assert config.grpo_profile is None
        assert config.grpo_rewards is None
        assert config.profile_config == {}


class TestRunConfig:
    def test_full_config(self):
        config = RunConfig(
            run=RunMetaConfig(method="sft"),
            pod=PodSpecConfig(),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
        )
        assert config.run.method == "sft"
        assert config.model.name == "Qwen/Qwen3-0.6B"

    def test_existing_pod(self):
        config = RunConfig(
            run=RunMetaConfig(method="sft", pod_id="pod_123"),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
        )
        assert config.run.pod_id == "pod_123"
        assert config.pod is None

    def test_no_pod_raises(self):
        with pytest.raises(Exception, match="pod_id"):
            RunConfig(
                run=RunMetaConfig(method="sft"),
                model=ModelConfig(name="Qwen/Qwen3-0.6B"),
                dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
            )

    def test_hub_cleanup_requires_push_to_hub(self):
        with pytest.raises(Exception, match="terminate_after_hub_push"):
            RunConfig(
                run=RunMetaConfig(method="sft", pod_id="pod_123", terminate_after_hub_push=True),
                model=ModelConfig(name="Qwen/Qwen3-0.6B"),
                dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
            )

    def test_hub_cleanup_allowed_when_push_to_hub_enabled(self):
        config = RunConfig(
            run=RunMetaConfig(method="sft", pod_id="pod_123", terminate_after_hub_push=True),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
            output=OutputConfig(push_to_hub=True, hub_model_id="user/model"),
        )
        assert config.run.terminate_after_hub_push is True
        assert config.output.push_to_hub is True

    def test_grpo_requires_profile(self):
        with pytest.raises(Exception, match="grpo_profile"):
            RunConfig(
                run=RunMetaConfig(method="grpo", pod_id="pod_123"),
                model=ModelConfig(name="Qwen/Qwen3-0.6B"),
                dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
            )

    def test_grpo_accepts_profile(self):
        config = RunConfig(
            run=RunMetaConfig(method="grpo", pod_id="pod_123"),
            model=ModelConfig(name="Qwen/Qwen3-0.6B"),
            dataset=DatasetConfig(source="huggingface", path="tatsu-lab/alpaca"),
            training=TrainingConfig(grpo_profile="ascii_art"),
        )
        assert config.training.grpo_profile == "ascii_art"

    def test_script_method_does_not_require_model_or_dataset(self):
        config = RunConfig(
            run=RunMetaConfig(method="script", pod_id="pod_123"),
            script=ScriptConfig(entrypoint="jobs/run.py"),
        )
        assert config.model is None
        assert config.dataset is None
        assert config.script.entrypoint == "jobs/run.py"

    def test_script_method_requires_entrypoint_or_command(self):
        with pytest.raises(Exception, match="script.entrypoint"):
            RunConfig(
                run=RunMetaConfig(method="script", pod_id="pod_123"),
            )

    def test_load_from_toml(self):
        toml_content = b"""
[run]
method = "sft"

[pod]
image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

[model]
name = "Qwen/Qwen3-0.6B"
adapter = "lora"
lora_r = 32

[dataset]
source = "huggingface"
path = "tatsu-lab/alpaca"
format = "alpaca"

[training]
epochs = 5
lr = 1e-4
batch_size = 8

[monitoring]
wandb_project = "test-project"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config = resolve_config(RunConfig, f.name)
        os.unlink(f.name)

        assert config.run.method == "sft"
        assert config.model.lora_r == 32
        assert config.training.epochs == 5
        assert config.training.lr == 1e-4
        assert config.monitoring.wandb_project == "test-project"

    def test_load_from_toml_with_utf8_bom(self):
        toml_content = (
            b"\xef\xbb\xbf"
            b"""
[run]
method = "sft"

[pod]
image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

[model]
name = "Qwen/Qwen3-0.6B"

[dataset]
source = "huggingface"
path = "tatsu-lab/alpaca"
validation_ratio = 0.1
"""
        )
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config = resolve_config(RunConfig, f.name)
        os.unlink(f.name)

        assert config.dataset.validation_ratio == 0.1

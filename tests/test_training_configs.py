from __future__ import annotations

import os
import tempfile

import pytest

from runpodx.configs.loader import resolve_config
from runpodx.training.configs import (
    DatasetConfig,
    ModelConfig,
    MonitoringConfig,
    OutputConfig,
    PodSpecConfig,
    RunConfig,
    RunMetaConfig,
    TrainingConfig,
)


class TestRunMetaConfig:
    def test_valid_method(self):
        config = RunMetaConfig(method="sft")
        assert config.method == "sft"

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


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 3
        assert config.lr == 2e-5
        assert config.bf16 is True


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

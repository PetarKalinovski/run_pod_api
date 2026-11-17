from __future__ import annotations

import os
import tempfile

import pytest

from runpodx.configs.loader import load_toml, resolve_config
from runpodx.configs.pod import GpuConfig, PodCreateConfig, StorageConfig
from runpodx.http.errors import ConfigError


class TestBaseConfig:
    def test_none_string_converted(self):
        config = GpuConfig(type_ids="None")
        assert config.type_ids is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            GpuConfig(nonexistent_field="value")


class TestPodCreateConfig:
    def test_minimal_config(self):
        config = PodCreateConfig(image="pytorch/pytorch:2.1.0")
        assert config.name == "my pod"
        assert config.gpu.count == 1
        assert config.storage.container_disk_gb == 50

    def test_full_config(self):
        config = PodCreateConfig(
            name="training",
            image="runpod/pytorch:2.1.0",
            cloud_type="SECURE",
            gpu=GpuConfig(count=4, type_ids=["NVIDIA H100 80GB HBM3"]),
            storage=StorageConfig(container_disk_gb=200, volume_gb=100),
            env={"WANDB_KEY": "test"},
        )
        assert config.gpu.count == 4
        assert config.storage.container_disk_gb == 200

    def test_to_api_payload_minimal(self):
        config = PodCreateConfig(image="pytorch/pytorch:2.1.0")
        payload = config.to_api_payload()
        assert payload["image"] == "pytorch/pytorch:2.1.0"
        assert payload["gpu"]["count"] == 1
        assert "id" not in payload["gpu"]
        assert payload["disk"] == 50
        assert payload["mounts"] == {"persistent": {"size": 20, "path": "/workspace"}}
        assert "gpuTypeIds" not in payload

    def test_to_api_payload_full(self):
        config = PodCreateConfig(
            name="test",
            image="pytorch/pytorch:2.1.0",
            gpu=GpuConfig(type_ids=["RTX 4090"], count=2),
            storage=StorageConfig(network_volume_id="vol_123"),
            docker_start_cmd=["python", "train.py"],
        )
        payload = config.to_api_payload()
        assert payload["gpu"]["id"] == "RTX 4090"
        assert payload["gpu"]["count"] == 2
        assert payload["mounts"] == {"network": [{"volumeId": "vol_123", "path": "/workspace"}]}
        assert payload["cmd"] == ["python", "train.py"]
        assert payload["cloud"] == "SECURE"

    def test_to_api_payload_one_gpu_type(self):
        config = PodCreateConfig(image="x", gpu=GpuConfig(type_ids=["A40", "L40S"]))
        assert config.to_api_payload()["gpu"]["id"] == "A40"
        assert config.to_api_payload(gpu_type_id="L40S")["gpu"]["id"] == "L40S"

    def test_to_api_payload_small_volume_raised_to_minimum(self):
        config = PodCreateConfig(image="x", storage=StorageConfig(volume_gb=5))
        assert config.to_api_payload()["mounts"]["persistent"]["size"] == 10

    def test_to_api_payload_no_volume(self):
        config = PodCreateConfig(image="x", storage=StorageConfig(volume_gb=0))
        assert "mounts" not in config.to_api_payload()


class TestTomlLoader:
    def test_load_toml_file(self):
        toml_content = b"""
name = "test-pod"
image = "pytorch/pytorch:2.1.0"

[gpu]
count = 2
type_ids = ["NVIDIA H100 80GB HBM3"]

[storage]
container_disk_gb = 100
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            data = load_toml(f.name)
        os.unlink(f.name)
        assert data["name"] == "test-pod"
        assert data["gpu"]["count"] == 2

    def test_load_nonexistent_file(self):
        with pytest.raises(ConfigError, match="not found"):
            load_toml("/nonexistent/path.toml")

    def test_load_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            f.flush()
            with pytest.raises(ConfigError, match=".toml"):
                load_toml(f.name)
        os.unlink(f.name)


class TestResolveConfig:
    def test_resolve_from_dict(self):
        config = resolve_config(PodCreateConfig, {"image": "test:latest"})
        assert isinstance(config, PodCreateConfig)
        assert config.image == "test:latest"

    def test_resolve_from_instance(self):
        original = PodCreateConfig(image="test:latest")
        resolved = resolve_config(PodCreateConfig, original)
        assert resolved is original

    def test_resolve_from_toml(self):
        toml_content = b'image = "test:latest"\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config = resolve_config(PodCreateConfig, f.name)
        os.unlink(f.name)
        assert config.image == "test:latest"

    def test_resolve_invalid_type(self):
        with pytest.raises(ConfigError):
            resolve_config(PodCreateConfig, 42)

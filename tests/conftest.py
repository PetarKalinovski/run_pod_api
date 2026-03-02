from __future__ import annotations

from typing import Any

import pytest

from runpodx.http.transport import HttpTransport


class MockTransport:
    """A mock transport that returns predefined responses without making HTTP calls."""

    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        self.calls.append((method, path, kwargs))
        key = f"{method} {path}"
        if key in self.responses:
            return self.responses[key]
        return self.responses.get(path, {})

    def close(self) -> None:
        pass


@pytest.fixture
def mock_transport():
    """Create a mock transport for testing."""
    return MockTransport


@pytest.fixture
def sample_pod_data():
    """Sample pod API response data."""
    return {
        "id": "pod_abc123",
        "name": "test-pod",
        "desiredStatus": "RUNNING",
        "imageName": "runpod/pytorch:2.1.0",
        "gpuTypeId": "NVIDIA GeForce RTX 4090",
        "gpuCount": 1,
        "containerDiskInGb": 50,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "publicIp": "1.2.3.4",
        "costPerHr": 0.44,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "dataCenterId": "US-TX-3",
        "portMappings": {"22": 15538, "8888": 18888},
    }

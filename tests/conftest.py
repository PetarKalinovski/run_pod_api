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
    """Sample pod API response data (REST v2's shape)."""
    return {
        "id": "pod_abc123",
        "name": "test-pod",
        "status": "RUNNING",
        "image": "runpod/pytorch:2.1.0",
        "disk": 50,
        "ports": ["22/tcp", "8888/http"],
        "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "vcpuCount": 16, "memory": 64},
        "mounts": {"persistent": {"size": 20, "path": "/workspace"}},
        "cloud": "SECURE",
        "dataCenterId": "US-TX-3",
        "cost": 0.44,
        "ssh": {"proxy": None, "direct": {"host": "1.2.3.4", "port": 15538, "username": "root",
                                          "command": "ssh root@1.2.3.4 -p 15538"}},
        "runtime": {"uptime": 120, "ports": [{"private": 22, "public": 15538, "type": "tcp", "ip": "1.2.3.4"},
                                             {"private": 8888, "public": 18888, "type": "tcp", "ip": "1.2.3.4"}]},
    }

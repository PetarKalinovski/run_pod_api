from __future__ import annotations

import pytest

from runpodx.http.errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    RunPodError,
    ServerError,
    ValidationError,
)
from runpodx.http.transport import HttpTransport, TransportConfig


class TestTransportConfig:
    def test_defaults(self):
        config = TransportConfig()
        assert config.base_url == "https://rest.runpod.io/v1"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_custom_values(self):
        config = TransportConfig(
            base_url="https://custom.api.io",
            api_key="test_key",
            timeout=60.0,
            max_retries=5,
        )
        assert config.base_url == "https://custom.api.io"
        assert config.api_key == "test_key"


class TestHttpTransport:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(AuthenticationError, match="No API key"):
            HttpTransport()

    def test_api_key_from_config(self):
        config = TransportConfig(api_key="test_key")
        transport = HttpTransport(config)
        assert transport._client.headers["authorization"] == "Bearer test_key"
        transport.close()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "env_key")
        transport = HttpTransport()
        assert transport._client.headers["authorization"] == "Bearer env_key"
        transport.close()

    def test_context_manager(self):
        config = TransportConfig(api_key="test_key")
        with HttpTransport(config) as transport:
            assert transport is not None


class TestErrorMapping:
    def test_error_hierarchy(self):
        assert issubclass(AuthenticationError, RunPodError)
        assert issubclass(NotFoundError, RunPodError)
        assert issubclass(ValidationError, RunPodError)
        assert issubclass(RateLimitError, RunPodError)
        assert issubclass(ServerError, RunPodError)

    def test_error_attributes(self):
        error = RunPodError("test", status_code=500, response_body={"error": "fail"})
        assert error.status_code == 500
        assert error.response_body == {"error": "fail"}
        assert str(error) == "test"

    def test_rate_limit_retry_after(self):
        error = RateLimitError("rate limited", retry_after=30.0, status_code=429)
        assert error.retry_after == 30.0
        assert error.status_code == 429

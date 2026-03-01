from __future__ import annotations


class RunPodError(Exception):
    """Base exception for all runpodx errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict | None = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class AuthenticationError(RunPodError):
    """Raised when the API key is missing or invalid (401)."""


class NotFoundError(RunPodError):
    """Raised when the requested resource does not exist (404)."""


class ValidationError(RunPodError):
    """Raised when the API rejects the request due to invalid input (400/422)."""


class RateLimitError(RunPodError):
    """Raised when rate-limited by the API (429)."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ServerError(RunPodError):
    """Raised when RunPod returns a 5xx error."""


class TimeoutError(RunPodError):
    """Raised when a request or polling operation times out."""


class ConfigError(RunPodError):
    """Raised when a TOML config file is invalid or cannot be loaded."""

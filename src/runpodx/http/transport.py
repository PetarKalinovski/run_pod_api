from __future__ import annotations

import logging
import os
from typing import Annotated, Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from runpodx.http.errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    RunPodError,
    ServerError,
    ValidationError,
)

logger = logging.getLogger("runpodx")

_BASE_URL = "https://rest.runpod.io/v1"


class TransportConfig(BaseModel):
    """Configuration for the HTTP transport layer."""

    model_config = ConfigDict(extra="forbid")

    base_url: Annotated[str, Field(description="RunPod REST API base URL.")] = _BASE_URL
    api_key: Annotated[
        str | None,
        Field(description="RunPod API key. Falls back to RUNPOD_API_KEY env var."),
    ] = None
    timeout: Annotated[float, Field(ge=1, description="Request timeout in seconds.")] = 30.0
    max_retries: Annotated[
        int, Field(ge=0, le=10, description="Maximum retry attempts for transient failures.")
    ] = 3


class HttpTransport:
    """Low-level HTTP transport with authentication, retry, and error mapping."""

    def __init__(self, config: TransportConfig | None = None) -> None:
        self._config = config or TransportConfig()
        api_key = self._config.api_key or os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "No API key provided. Set RUNPOD_API_KEY or pass api_key to RunPodClient."
            )
        self._client = httpx.Client(
            base_url=self._config.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout,
        )
        self._max_retries = self._config.max_retries

    def request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        """Execute an HTTP request with retry and error handling."""
        return self._request_with_retry(method, path, **kwargs)

    @retry(
        retry=retry_if_exception_type((ServerError, RateLimitError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _request_with_retry(self, method: str, path: str, **kwargs: Any) -> dict | list:
        logger.debug("%s %s %s", method, path, kwargs.get("params", ""))
        response = self._client.request(method, path, **kwargs)
        if response.is_success:
            if response.status_code == 204:
                return {}
            return response.json()
        self._raise_for_status(response)

    def _raise_for_status(self, response: httpx.Response) -> None:
        body = None
        try:
            body = response.json()
        except Exception:
            pass
        status = response.status_code
        msg = f"HTTP {status}: {body or response.text}"

        if status == 401:
            raise AuthenticationError(msg, status_code=status, response_body=body)
        elif status == 404:
            raise NotFoundError(msg, status_code=status, response_body=body)
        elif status in (400, 422):
            raise ValidationError(msg, status_code=status, response_body=body)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                msg,
                status_code=status,
                response_body=body,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif status >= 500:
            raise ServerError(msg, status_code=status, response_body=body)
        else:
            raise RunPodError(msg, status_code=status, response_body=body)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

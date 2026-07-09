"""Base classes for LLM API clients."""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests


class ProviderRequestError(RuntimeError):
    """Normalized provider failure with the consumed retry count."""

    def __init__(self, message: str, *, attempts: int, error_type: str):
        super().__init__(message)
        self.attempts = attempts
        self.error_type = error_type


class ProviderResponse(str):
    """String-compatible model response with provider metadata."""

    def __new__(
        cls,
        text: str,
        *,
        finish_reason: str | None = None,
        usage: Dict[str, Any] | None = None,
        response_id: str | None = None,
        actual_model: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        instance = super().__new__(cls, text)
        instance.text = text
        instance.finish_reason = finish_reason
        instance.usage = dict(usage or {})
        instance.response_id = response_id
        instance.actual_model = actual_model
        instance.metadata = dict(metadata or {})
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable metadata without duplicating the response text."""
        return {
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "response_id": self.response_id,
            "actual_model": self.actual_model,
            "metadata": self.metadata,
        }


class APIClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    @abstractmethod
    def query(
        self,
        prompt: str,
        max_tokens: int = 1024,
        retries: int = 3,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> ProviderResponse:
        """Query the LLM API with retry logic."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict]:
        """List available models."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if API is accessible."""
        pass

    def close(self) -> None:
        """Close any persistent client resources."""
        return None

    def get_model_metadata(self) -> Dict[str, Any]:
        """Resolve the requested model to provider metadata when available."""
        base_metadata = {
            "provider_client": type(self).__name__,
            "endpoint": self.base_url,
            "provider_version": "unknown",
            "provider_version_unavailable_reason": (
                "the configured model-listing API does not expose server version"
            ),
        }
        try:
            models = self.list_models()
        except Exception as exc:  # noqa: BLE001 - provenance must not abort a run.
            return {
                **base_metadata,
                "requested_model": self.model_name,
                "status": "unavailable",
                "reason": str(exc),
            }

        for model in models:
            candidate = model.get("id") or model.get("name") or model.get("model")
            if candidate == self.model_name:
                return {
                    **base_metadata,
                    "requested_model": self.model_name,
                    "status": "resolved",
                    "model": model,
                }
        return {
            **base_metadata,
            "requested_model": self.model_name,
            "status": "not_found",
            "reason": "provider model listing did not contain the requested model",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class RequestsRetryMixin:
    """Shared retry/error handling for requests-based model clients."""

    provider_name: str
    base_url: str
    session: requests.Session
    timeout: int

    def _post_json_with_retries(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        retries: int,
    ) -> Dict[str, Any]:
        """POST JSON with retry behavior used by local providers."""
        for attempt in range(retries):
            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                print(f"   Timeout on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise ProviderRequestError(
                        f"API timeout after {retries} attempts",
                        attempts=attempt + 1,
                        error_type="timeout",
                    ) from None
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError as e:
                raise ProviderRequestError(
                    f"Cannot connect to {self.provider_name} at {self.base_url}. "
                    "Is it running?",
                    attempts=attempt + 1,
                    error_type="connection",
                ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("   Rate limited, waiting...")
                    if attempt == retries - 1:
                        raise ProviderRequestError(
                            "Rate limit retry budget exhausted",
                            attempts=attempt + 1,
                            error_type="rate_limit",
                        ) from e
                    time.sleep(5)
                    continue
                raise ProviderRequestError(
                    f"API error {e.response.status_code}: {e.response.text}",
                    attempts=attempt + 1,
                    error_type="http",
                ) from e

            except (KeyError, json.JSONDecodeError) as e:
                raise ProviderRequestError(
                    f"Invalid API response format: {e}",
                    attempts=attempt + 1,
                    error_type="invalid_response",
                ) from e

        raise ProviderRequestError(
            "Max retries exceeded", attempts=retries, error_type="retry_exhausted"
        )

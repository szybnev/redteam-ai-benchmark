"""OpenWebUI API client (OpenAI-compatible with optional authentication)."""

import json
import os
import time
from typing import Dict, List, Optional

import requests

from .base import APIClient, ProviderRequestError, ProviderResponse


class OpenWebUIClient(APIClient):
    """
    OpenWebUI API client.

    OpenWebUI provides an OpenAI-compatible API with optional Bearer token authentication.
    Supports both local instances (no auth) and secured deployments (Bearer token).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        model_name: str = "",
        api_key: Optional[str] = None,
        timeout: int = 150,
    ):
        """
        Initialize OpenWebUI client.

        Args:
            base_url: OpenWebUI instance URL (default: http://localhost:3000)
            model_name: Model name/ID to use
            api_key: Optional API key for authentication (or set OPENWEBUI_API_KEY env var)
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, model_name)
        self.api_key = api_key or os.environ.get("OPENWEBUI_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional authentication."""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def query(
        self,
        prompt: str,
        max_tokens: int = 768,
        retries: int = 3,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> ProviderResponse:
        """Query OpenWebUI API with retry logic."""

        url = f"{self.base_url}/api/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed

        for attempt in range(retries):
            try:
                response = self.session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                return ProviderResponse(
                    choice["message"]["content"],
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage"),
                    response_id=data.get("id"),
                    actual_model=data.get("model", self.model_name),
                )

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
                    f"Cannot connect to OpenWebUI at {self.base_url}. Is it running?",
                    attempts=attempt + 1,
                    error_type="connection",
                ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    raise RuntimeError(
                        "Authentication required. Provide API key via --api-key or OPENWEBUI_API_KEY"
                    ) from e
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

    def list_models(self) -> List[Dict]:
        """List available models from OpenWebUI."""
        try:
            url = f"{self.base_url}/api/models"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
            # OpenWebUI returns models in "data" array (OpenAI format)
            return data.get("data", data.get("models", []))

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    "Authentication required. Provide API key via --api-key or OPENWEBUI_API_KEY"
                ) from e
            raise RuntimeError(f"Failed to list models: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test OpenWebUI connection."""
        try:
            url = f"{self.base_url}/api/models"
            response = self.session.get(url, headers=self._get_headers(), timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()

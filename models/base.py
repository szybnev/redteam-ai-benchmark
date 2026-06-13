"""Base classes for LLM API clients."""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests


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
    ) -> str:
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
                    raise RuntimeError(
                        f"API timeout after {retries} attempts"
                    ) from None
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to {self.provider_name} at {self.base_url}. "
                    "Is it running?"
                ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("   Rate limited, waiting...")
                    time.sleep(5)
                    continue
                raise RuntimeError(
                    f"API error {e.response.status_code}: {e.response.text}"
                ) from e

            except (KeyError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Invalid API response format: {e}") from e

        raise RuntimeError("Max retries exceeded")

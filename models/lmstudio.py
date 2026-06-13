"""LM Studio API client (OpenAI-compatible)."""

from typing import Dict, List

import requests

from .base import APIClient, RequestsRetryMixin


class LMStudioClient(RequestsRetryMixin, APIClient):
    """LM Studio API client (OpenAI-compatible)."""

    provider_name = "LM Studio"

    def __init__(self, base_url: str, model_name: str, timeout: int = 150):
        super().__init__(base_url, model_name)
        self.timeout = timeout
        self.session = requests.Session()

    def query(
        self,
        prompt: str,
        max_tokens: int = 768,
        retries: int = 3,
        temperature: float = 0.2,
    ) -> str:
        """Query LM Studio API with retry logic."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = self._post_json_with_retries(
            url=url, headers=headers, payload=payload, retries=retries
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """List available models from LM Studio."""
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()

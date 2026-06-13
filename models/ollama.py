"""Ollama API client."""

from typing import Dict, List

import requests

from .base import APIClient, RequestsRetryMixin


class OllamaClient(RequestsRetryMixin, APIClient):
    """Ollama API client."""

    provider_name = "Ollama"

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
        """Query Ollama API with retry logic."""
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = self._post_json_with_retries(
            url=url, headers=headers, payload=payload, retries=retries
        )
        try:
            return data["message"]["content"]
        except KeyError as e:
            raise RuntimeError(f"Invalid API response format: {e}") from e

    def list_models(self) -> List[Dict]:
        """List available models from Ollama."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}") from e

    def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the persistent HTTP session."""
        self.session.close()

"""Helpers for loading embedding models consistently."""

import os
from typing import Any, Dict

from .constants import DEFAULT_SEMANTIC_MODEL

SEMANTIC_DEVICE_ENV = "REDTEAM_SEMANTIC_DEVICE"


def sentence_transformer_kwargs(model_name: str) -> Dict[str, Any]:
    """Return model-specific SentenceTransformer initialization kwargs."""
    lower_name = model_name.lower()

    if "gte" in lower_name or "alibaba" in lower_name:
        return {"trust_remote_code": True}

    if model_name == DEFAULT_SEMANTIC_MODEL or lower_name.startswith("qwen/qwen3-embedding"):
        return {"device": os.environ.get(SEMANTIC_DEVICE_ENV, "cpu")}

    return {}

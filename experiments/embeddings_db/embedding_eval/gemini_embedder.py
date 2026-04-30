import math
import time
from dataclasses import dataclass
from typing import Optional

import requests


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class EmbeddingError(Exception):
    pass


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    model: str
    embedding_task_setup: str
    output_dimensionality: int
    doc_text_variant: str
    query_text_variant: str
    similarity_metric: str
    normalize_embeddings: bool
    query_task_type: Optional[str] = None
    doc_task_type: Optional[str] = None


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 60,
        max_retries: int = 5,
        initial_backoff_seconds: float = 1.0,
    ):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.session = requests.Session()

    def list_models(self) -> list[str]:
        url = f"{GEMINI_API_BASE}/models"
        response = self.session.get(
            url,
            params={"key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return [model.get("name", "").split("/")[-1] for model in payload.get("models", [])]

    def supports_model(self, model: str) -> bool:
        return model in self.list_models()

    def embed_text(
        self,
        model: str,
        text: str,
        output_dimensionality: int,
        task_type: Optional[str],
    ) -> list[float]:
        body = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": output_dimensionality,
        }
        if task_type is not None:
            body["taskType"] = task_type
        url = f"{GEMINI_API_BASE}/models/{model}:embedContent"

        for attempt in range(self.max_retries):
            response = self.session.post(
                url,
                params={"key": self.api_key},
                json=body,
                timeout=self.timeout_seconds,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                backoff = self.initial_backoff_seconds * (2 ** attempt)
                time.sleep(backoff)
                continue
            if not response.ok:
                raise EmbeddingError(f"Embedding call failed: {response.status_code} {response.text}")
            payload = response.json()
            embedding = payload.get("embedding") or {}
            values = embedding.get("values")
            if values is None:
                raise EmbeddingError(f"Missing embedding values in response: {payload}")
            return values

        raise EmbeddingError(f"Embedding call failed after {self.max_retries} retries for model={model}")


def l2_normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

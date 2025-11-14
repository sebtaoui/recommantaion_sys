from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Union

import numpy as np
import httpx

from sentence_transformers import SentenceTransformer
from openai import OpenAI


class EmbeddingBackend:
    """
    Interface minimale attendue par le pipeline : encode(texts, normalize).
    """

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> Union[np.ndarray, List[float]]:
        raise NotImplementedError


class LocalSentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model = SentenceTransformer(model_name)
        if device:
            self._model = self._model.to(device)

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> Union[np.ndarray, List[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )


class TogetherEmbeddingBackend(EmbeddingBackend):
    """
    Encodage via Together.ai (API OpenAI compatible).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> None:
        # Configuration des timeouts pour éviter les erreurs de connexion
        _timeout = httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s pour la connexion
        _http_client = httpx.Client(timeout=_timeout, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=_http_client,
            max_retries=2
        )
        self._model_name = model_name

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        normalize: bool = True,
    ) -> Union[np.ndarray, List[float]]:
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=inputs,
            )
            vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]

            if normalize:
                vectors = [self._normalize(vector) for vector in vectors]

            if single:
                return vectors[0]
            return np.stack(vectors)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, ConnectionResetError) as e:
            print(f"[Embedding Error] Connection error: {e}")
            # Retourner un vecteur zéro de la dimension attendue (approximation)
            # En production, vous pourriez vouloir lever une exception ou utiliser un fallback
            dim = 768  # Dimension par défaut pour la plupart des modèles
            zero_vector = np.zeros(dim, dtype=np.float32)
            if normalize:
                zero_vector = self._normalize(zero_vector)
            if single:
                return zero_vector
            return np.stack([zero_vector] * len(inputs))
        except Exception as e:
            print(f"[Embedding Error] Unexpected error: {e}")
            # Même fallback
            dim = 768
            zero_vector = np.zeros(dim, dtype=np.float32)
            if normalize:
                zero_vector = self._normalize(zero_vector)
            if single:
                return zero_vector
            return np.stack([zero_vector] * len(inputs))

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm


def build_embedding_backend(
    provider: str,
    *,
    model_name: str,
    together_api_key: str | None = None,
    together_base_url: str | None = None,
    device: str | None = None,
) -> EmbeddingBackend:
    provider = (provider or "local").lower()

    if provider == "together":
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY must be set when using Together embeddings.")
        base_url = together_base_url or "https://api.together.xyz/v1"
        return TogetherEmbeddingBackend(
            api_key=together_api_key,
            base_url=base_url,
            model_name=model_name,
        )

    return LocalSentenceTransformerBackend(model_name=model_name, device=device)



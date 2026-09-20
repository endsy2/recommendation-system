"""Model loading and memory-conscious lyric embedding helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .chunking import content_token_ids, model_inputs_for_content, special_token_count
from .config import EMBEDDING_DIMENSION, MODEL_ID


class EmbeddingError(RuntimeError):
    pass


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise EmbeddingError("Embedding is not a finite one-dimensional vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.isfinite(norm):
        raise EmbeddingError("Embedding has zero or invalid L2 norm")
    return value / norm


def pool_normalized_chunks(chunk_embeddings: Sequence[np.ndarray], expected_dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    """Normalize chunks, mean-pool only one song's chunks, then normalize the mean."""
    if not chunk_embeddings:
        raise EmbeddingError("Cannot pool zero chunk embeddings")
    total = np.zeros(expected_dimension, dtype=np.float64)
    for embedding in chunk_embeddings:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (expected_dimension,):
            raise EmbeddingError(f"Expected vector shape ({expected_dimension},), got {vector.shape}")
        total += l2_normalize(vector)
    return l2_normalize((total / len(chunk_embeddings)).astype(np.float32))


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str | None
    dimension: int
    input_limit: int
    special_tokens: int


class SentenceTransformerEmbedder:
    """Uses exact token IDs for chunks so tokenizer truncation cannot be hidden."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers is required; install requirements.txt") from exc
        self.model = SentenceTransformer(model_id, device=device)
        self.tokenizer = self.model.tokenizer
        self.info = ModelInfo(
            model_id=model_id,
            revision=self._resolved_revision(),
            dimension=int(self.model.get_sentence_embedding_dimension()),
            input_limit=self._input_limit(),
            special_tokens=special_token_count(self.tokenizer),
        )
        if self.info.dimension != EMBEDDING_DIMENSION:
            raise EmbeddingError(
                f"Model reports {self.info.dimension} dimensions, expected {EMBEDDING_DIMENSION}. "
                "Use a compatible all-MiniLM-L6-v2 model."
            )

    def _input_limit(self) -> int:
        candidates: list[int] = []
        for value in (getattr(self.model, "max_seq_length", None), getattr(self.tokenizer, "model_max_length", None)):
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)
        first_module = self.model._first_module()
        auto_model = getattr(first_module, "auto_model", None)
        position_limit = getattr(getattr(auto_model, "config", None), "max_position_embeddings", None)
        if isinstance(position_limit, int) and position_limit > 0:
            candidates.append(position_limit)
        if not candidates:
            raise EmbeddingError("Could not determine the loaded model input limit")
        return min(candidates)

    def _resolved_revision(self) -> str | None:
        # Some cached models expose a commit hash. Absence is recorded as null, never guessed.
        config = getattr(getattr(self.model._first_module(), "auto_model", None), "config", None)
        revision = getattr(config, "_commit_hash", None)
        return revision if isinstance(revision, str) and revision else None

    def tokenize(self, text: str) -> list[int]:
        return content_token_ids(self.tokenizer, text)

    def encode_content_batches(self, chunks: Sequence[Sequence[int]], batch_size: int) -> Iterable[np.ndarray]:
        """Yield normalized vectors in order, in bounded batches of pre-tokenized inputs."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            import torch
        except ImportError as exc:
            raise EmbeddingError("PyTorch is required by sentence-transformers") from exc
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            features = [model_inputs_for_content(self.tokenizer, ids, self.info.input_limit) for ids in batch]
            padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            device = self.model.device
            padded = {name: value.to(device) for name, value in padded.items()}
            with torch.no_grad():
                output = self.model(padded)["sentence_embedding"].detach().cpu().numpy()
            for row in output:
                yield l2_normalize(np.asarray(row, dtype=np.float32))

    def encode_query(self, query: str) -> np.ndarray:
        ids = self.tokenize(query)
        # A query must fit as one model input; callers intentionally reject oversized requests.
        return next(self.encode_content_batches([ids], batch_size=1))


"""Reusable lyrics-only retrieval over a validated Milvus collection (HNSW index, COSINE metric)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import EMBEDDING_DIMENSION, MilvusSettings
from .embeddings import SentenceTransformerEmbedder
from .milvus_store import (
    CollectionNotReadyError,
    IncompatibleCollectionError,
    MilvusSongStore,
    NotReadyError,
    StoredHit,
)
from .preprocess import safe_source_url

# Kept as the name the app/tests catch for "cannot serve"; covers Milvus and model readiness.
ArtifactError = NotReadyError


class QueryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SearchResult:
    rank: int
    song_id: str
    artist: str
    song: str
    similarity_score: float
    source_link: str | None
    safe_source_url: str | None
    excerpt: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "song_id": self.song_id, "artist": self.artist, "song": self.song,
                "similarity_score": round(self.similarity_score, 6), "source_link": self.source_link,
                "safe_source_url": self.safe_source_url, "excerpt": self.excerpt}

    @classmethod
    def from_hit(cls, rank: int, hit: StoredHit) -> "SearchResult":
        # Raw cosine similarity; never rescaled into a percentage or probability.
        return cls(rank, hit.song_id, hit.artist, hit.song, hit.score, hit.link, safe_source_url(hit.link),
                   hit.lyrics_excerpt)


def check_embedder_compatibility(manifest: dict[str, Any], embedder: Any) -> None:
    model = manifest.get("model") or {}
    info = embedder.info
    if getattr(info, "model_id", model.get("identifier")) != model.get("identifier"):
        raise IncompatibleCollectionError("The loaded embedding model differs from the one used to build the collection")
    if info.dimension != manifest["embedding_dimension"]:
        raise IncompatibleCollectionError("Loaded model dimension is incompatible with the collection")
    if info.input_limit < manifest["tokenizer_model_input_limit"]:
        raise IncompatibleCollectionError("Loaded model input limit is smaller than the collection configuration")
    built, loaded = model.get("resolved_revision"), getattr(info, "revision", None)
    if built and loaded and built != loaded:
        raise IncompatibleCollectionError(
            f"Loaded model revision {loaded} differs from build revision {built}; rebuild into a new collection"
        )


class SearchEngine:
    def __init__(self, store: Any, manifest: dict[str, Any], embedder: Any) -> None:
        self.store, self.manifest, self.embedder = store, manifest, embedder
        self.last_metrics: dict[str, float] = {}

    @classmethod
    def from_milvus(
        cls,
        settings: MilvusSettings,
        device: str | None = None,
        embedder: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> "SearchEngine":
        store = MilvusSongStore.open(settings, client_factory)
        try:
            if embedder is None:
                try:
                    embedder = SentenceTransformerEmbedder(store.manifest["model"]["identifier"], device=device)
                except Exception as exc:
                    raise CollectionNotReadyError(
                        f"Could not load the embedding model required by this collection ({type(exc).__name__})"
                    ) from exc
            check_embedder_compatibility(store.manifest, embedder)
        except BaseException:
            store.close()
            raise
        return cls(store, store.manifest, embedder)

    @property
    def collection(self) -> str:
        return self.store.collection

    @property
    def song_count(self) -> int:
        return int(self.manifest.get("imported_song_count", 0))

    def close(self) -> None:
        self.store.close()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        return self.search_with_metrics(query, top_k)[0]

    def search_with_metrics(self, query: str, top_k: int = 10) -> tuple[list[SearchResult], dict[str, float]]:
        if not isinstance(query, str) or not query.strip():
            raise QueryValidationError("Query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise QueryValidationError("top_k must be an integer from 1 to 50")
        query = query.strip()
        content_ids = self.embedder.tokenize(query)
        limit = min(self.manifest["tokenizer_model_input_limit"], self.embedder.info.input_limit)
        if len(content_ids) + self.manifest["special_tokens_per_input"] > limit:
            raise QueryValidationError(f"Query is too long ({len(content_ids)} content tokens); it must fit within "
                                       f"{limit - self.manifest['special_tokens_per_input']} content tokens.")
        started = time.perf_counter()
        vector = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        embedding_seconds = time.perf_counter() - started
        if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
            raise QueryValidationError("Query embedding is invalid")
        search_started = time.perf_counter()
        # Milvus returns fewer hits when the collection holds fewer songs than top_k.
        hits = self.store.search(vector, top_k)[:top_k]
        search_seconds = time.perf_counter() - search_started
        results = [SearchResult.from_hit(rank, hit) for rank, hit in enumerate(hits, start=1)]
        metrics = {"query_embedding_seconds": embedding_seconds, "vector_search_seconds": search_seconds,
                   "end_to_end_seconds": time.perf_counter() - started}
        self.last_metrics = metrics
        return results, metrics

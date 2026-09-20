"""Synthetic rows, manifests, and FAISS artifacts shared by unit and integration tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import EMBEDDING_DIMENSION, MODEL_ID, POOLING_DESCRIPTION
from src.milvus_store import SongRow, new_manifest


def unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, EMBEDDING_DIMENSION)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def make_rows(count: int, seed: int = 0, first_id: int = 100) -> list[SongRow]:
    vectors = unit_vectors(count, seed)
    return [SongRow(str(first_id + i), vectors[i], f"Artist {i}", f"Song {i}",
                    None if i % 3 == 0 else f"/a/artist/song_{i}.html", f"Lyrics excerpt {i}…")
            for i in range(count)]


def make_manifest(collection: str, count: int, build_mode: str = "sample", fingerprint: str = "sha256:abc",
                  revision: str | None = "rev-1") -> dict:
    return new_manifest(
        collection, model={"identifier": MODEL_ID, "resolved_revision": revision}, tokenizer_model_input_limit=256,
        special_tokens_per_input=2, chunking={"content_token_chunk_size": 200, "overlap": 32, "stride": 168},
        pooling=POOLING_DESCRIPTION,
        preprocessing={"original_rows": 10, "missing_or_empty_lyrics": 0, "duplicate_rows_removed": 0,
                       "valid_cleaned_count": 10, "selected_for_build": count},
        dataset_fingerprint=fingerprint, build_mode=build_mode, sample_seed=42 if build_mode == "sample" else None,
        expected_song_count=count, excluded=[], embeddings="test", source={"type": "test"}, dependency_versions={},
    )


def batches_of(rows: list[SongRow], size: int):
    return lambda: (rows[start:start + size] for start in range(0, len(rows), size))


def write_faiss_artifact(directory: Path, rows: list[SongRow], index_factory=None, build_mode: str = "sample") -> Path:
    """Write an artifact in the exact layout the former FAISS build_index produced."""
    import faiss

    directory.mkdir(parents=True, exist_ok=True)
    index = index_factory() if index_factory else faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    matrix = np.vstack([row.vector for row in rows]).astype(np.float32)
    if not index.is_trained:
        index.train(matrix)
    index.add(matrix)
    faiss.write_index(index, str(directory / "songs.faiss"))
    metadata = [{"song_id": row.song_id, "artist": row.artist, "song": row.song, "source_link": row.link,
                 "safe_source_url": None, "excerpt": row.lyrics_excerpt} for row in rows]
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "format_version": 1, "model": {"identifier": MODEL_ID, "resolved_revision": "rev-1"},
        "embedding_dimension": EMBEDDING_DIMENSION, "tokenizer_model_input_limit": 256, "special_tokens_per_input": 2,
        "chunking": {"content_token_chunk_size": 200, "overlap": 32, "stride": 168}, "pooling": POOLING_DESCRIPTION,
        "similarity": "cosine similarity via inner product of L2-normalized vectors", "indexed_song_count": len(rows),
        "excluded_invalid_vectors": 0,
        "preprocessing": {"original_rows": 10, "missing_or_empty_lyrics": 0, "duplicate_rows_removed": 0,
                          "valid_cleaned_count": 10, "selected_for_build": len(rows)},
        "dataset_fingerprint": "sha256:abc", "build_mode": build_mode, "sample_seed": 42 if build_mode == "sample" else None,
        "faiss_index_file_bytes": (directory / "songs.faiss").stat().st_size,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory

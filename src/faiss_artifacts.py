"""Read-only access to trusted, project-generated FAISS artifacts (migration and baseline checks only).

Only artifacts written by this project's former ``build_index`` (manifest ``format_version`` 1) are
accepted. ``faiss.read_index`` parses a binary format, so the manifest and file size are checked
before the index is opened. Nothing here writes to or deletes the source files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import EMBEDDING_DIMENSION, MODEL_ID, POOLING_DESCRIPTION
from .milvus_store import SongRow, duplicate_ids, row_from_metadata, row_problems
from .preprocess import dataset_fingerprint, load_and_clean_csv


class ArtifactExportError(RuntimeError):
    """The artifact cannot be exported safely; nothing was written anywhere."""


@dataclass
class FaissArtifact:
    directory: Path
    index_path: Path
    index: Any
    index_type: str
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @property
    def song_ids(self) -> list[str]:
        return [str(item["song_id"]) for item in self.metadata]

    def iter_vector_batches(self, batch_size: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (start_position, float32 matrix) for exact stored vectors, in manageable batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.count, batch_size):
            count = min(batch_size, self.count - start)
            block = np.asarray(self.index.reconstruct_n(start, count), dtype=np.float32)
            if block.shape != (count, EMBEDDING_DIMENSION):
                raise ArtifactExportError(f"Unexpected reconstructed block shape {block.shape} at position {start}")
            yield start, block

    def iter_rows(self, batch_size: int) -> Iterator[list[SongRow]]:
        """FAISS position i pairs with metadata.json position i (how build_index wrote them)."""
        for start, block in self.iter_vector_batches(batch_size):
            yield [row_from_metadata(self.metadata[start + offset], block[offset]) for offset in range(len(block))]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactExportError(f"Could not read {label} {path}: {exc}") from exc


def manifest_problems(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest.json must be an object"]
    problems = []
    if manifest.get("format_version") != 1:
        problems.append("manifest format_version is not 1 (not a project-generated artifact)")
    if (manifest.get("model") or {}).get("identifier") != MODEL_ID:
        problems.append(f"model is {(manifest.get('model') or {}).get('identifier')!r}, expected {MODEL_ID!r}")
    if manifest.get("embedding_dimension") != EMBEDDING_DIMENSION:
        problems.append(f"embedding_dimension is {manifest.get('embedding_dimension')}, expected {EMBEDDING_DIMENSION}")
    if manifest.get("pooling") != POOLING_DESCRIPTION:
        problems.append("pooling description does not match normalized-chunk mean pooling")
    chunking = manifest.get("chunking") or {}
    size, overlap, stride = (chunking.get(key) for key in ("content_token_chunk_size", "overlap", "stride"))
    if not all(isinstance(value, int) for value in (size, overlap, stride)) or stride != size - overlap:
        problems.append("chunking settings are missing or inconsistent")
    if manifest.get("build_mode") not in ("sample", "full"):
        problems.append("build_mode must be 'sample' or 'full'")
    if not str(manifest.get("dataset_fingerprint", "")).startswith("sha256:"):
        problems.append("dataset_fingerprint is missing")
    for key in ("indexed_song_count", "tokenizer_model_input_limit", "special_tokens_per_input"):
        if not isinstance(manifest.get(key), int):
            problems.append(f"{key} is missing")
    if not isinstance(manifest.get("preprocessing"), dict):
        problems.append("preprocessing report is missing")
    return problems


def load_faiss_artifact(
    directory: str | Path,
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> FaissArtifact:
    base = Path(directory)
    index_file = Path(index_path) if index_path else base / "songs.faiss"
    metadata_file = Path(metadata_path) if metadata_path else base / "metadata.json"
    manifest_file = Path(manifest_path) if manifest_path else base / "manifest.json"
    missing = [str(path) for path in (index_file, metadata_file, manifest_file) if not path.is_file()]
    if missing:
        raise ArtifactExportError(f"Artifact is incomplete; missing: {', '.join(missing)}. "
                                  "Build from the CSV instead: python -m src.build_index ...")
    manifest = _read_json(manifest_file, "manifest")
    problems = manifest_problems(manifest)
    if problems:
        raise ArtifactExportError(f"{manifest_file} is not a trusted project artifact: " + "; ".join(problems))
    expected_bytes = manifest.get("faiss_index_file_bytes")
    if expected_bytes != index_file.stat().st_size:
        raise ArtifactExportError(f"{index_file} is {index_file.stat().st_size} bytes but its manifest recorded "
                                  f"{expected_bytes}; refusing to read a file that does not match its build record.")
    metadata = _read_json(metadata_file, "metadata")
    if not isinstance(metadata, list):
        raise ArtifactExportError("metadata.json must be a list")

    try:
        import faiss
    except ImportError as exc:
        raise ArtifactExportError("faiss-cpu is required to read FAISS artifacts; install requirements.txt") from exc
    try:
        # read_index returns the concrete index class. Do not wrap it in downcast_index(): the
        # temporary returned by read_index owns the C++ object, and letting it be garbage-collected
        # leaves the downcast wrapper dangling (reconstruct_n then aborts the process).
        index = faiss.read_index(str(index_file))
    except Exception as exc:
        raise ArtifactExportError(f"Could not read FAISS index {index_file}: {exc}") from exc
    index_type = type(index).__name__
    # Only an exact IndexFlatIP stores raw vectors that reconstruct_n returns losslessly. Other index
    # types may quantize, compress, or not support reconstruction at all.
    if type(index) is not faiss.IndexFlatIP or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ArtifactExportError(f"Unsupported FAISS index type {index_type} (metric {index.metric_type}); safe "
                                  "export is implemented only for IndexFlatIP. Rebuild from the CSV instead.")
    if index.d != EMBEDDING_DIMENSION:
        raise ArtifactExportError(f"FAISS index dimension is {index.d}, expected {EMBEDDING_DIMENSION}")
    if not (index.ntotal == len(metadata) == manifest["indexed_song_count"]):
        raise ArtifactExportError(f"Count mismatch: FAISS={index.ntotal}, metadata={len(metadata)}, "
                                  f"manifest={manifest['indexed_song_count']}")
    for position, item in enumerate(metadata):
        if not isinstance(item, dict) or not all(key in item for key in ("song_id", "artist", "song")):
            raise ArtifactExportError(f"Invalid metadata entry at position {position}")
    return FaissArtifact(base, index_file, index, index_type, metadata, manifest)


def validate_artifact_rows(artifact: FaissArtifact, batch_size: int) -> dict[str, Any]:
    """Check every vector and metadata record before anything is written. Raises on any problem."""
    duplicates = duplicate_ids(artifact.song_ids)
    if duplicates:
        raise ArtifactExportError(f"Duplicate song IDs in metadata: {', '.join(duplicates[:20])}")
    problems: list[str] = []
    min_norm, max_norm = float("inf"), 0.0
    for batch in artifact.iter_rows(batch_size):
        for row in batch:
            for problem in row_problems(row):
                problems.append(f"song {row.song_id}: {problem}")
            if np.isfinite(row.vector).all():
                norm = float(np.linalg.norm(row.vector.astype(np.float64)))
                min_norm, max_norm = min(min_norm, norm), max(max_norm, norm)
    if problems:
        shown = "; ".join(problems[:20])
        raise ArtifactExportError(f"{len(problems)} record problem(s); nothing was written. First: {shown}")
    return {"records": artifact.count, "min_vector_norm": min_norm, "max_vector_norm": max_norm}


def verify_against_csv(artifact: FaissArtifact, csv_path: str | Path) -> dict[str, Any]:
    """Confirm the artifact came from this CSV and that each song_id still maps to the same song."""
    fingerprint = dataset_fingerprint(Path(csv_path))
    if fingerprint != artifact.manifest["dataset_fingerprint"]:
        raise ArtifactExportError(f"CSV fingerprint {fingerprint} does not match the artifact's "
                                  f"{artifact.manifest['dataset_fingerprint']}")
    records, report = load_and_clean_csv(csv_path)
    recorded = artifact.manifest["preprocessing"]
    for key in ("original_rows", "missing_or_empty_lyrics", "duplicate_rows_removed", "valid_cleaned_count"):
        if recorded.get(key) != getattr(report, key):
            raise ArtifactExportError(f"Preprocessing {key} is {getattr(report, key)} now but was {recorded.get(key)}")
    by_id = {record.song_id: record for record in records}
    mismatched = [str(item["song_id"]) for item in artifact.metadata
                  if str(item["song_id"]) not in by_id or by_id[str(item["song_id"])].metadata() != item]
    if mismatched:
        raise ArtifactExportError(f"{len(mismatched)} metadata record(s) no longer match the cleaned CSV "
                                  f"(first: {', '.join(mismatched[:10])})")
    return {"csv_fingerprint_matches": True, "preprocessing_counts_match": True, "metadata_matches_csv": artifact.count}

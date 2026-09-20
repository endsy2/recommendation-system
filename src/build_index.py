"""CLI to preprocess lyrics, build one vector/song, and import them into a Milvus collection."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import ChunkConfig, chunk_token_ids
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    EMBEDDING_DIMENSION,
    POOLING_DESCRIPTION,
    ConfigError,
    MilvusSettings,
    default_csv_path,
    load_env_file,
)
from .embeddings import EmbeddingError, SentenceTransformerEmbedder, l2_normalize
from .milvus_store import (
    MAX_BATCH_SIZE,
    NotReadyError,
    RecordValidationError,
    SongRow,
    check_collection_matches_build_mode,
    connect,
    dependency_versions,
    import_rows,
    new_manifest,
    prepare_destination,
    row_problems,
)
from .preprocess import SongRecord, dataset_fingerprint, load_and_clean_csv, safe_source_url, sample_records


class BuildError(RuntimeError):
    pass


def _memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _atomic_publish(temp_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise BuildError(f"Output directory exists: {output_dir}. Use --overwrite only after checking it.")
        backup = output_dir.with_name(f".{output_dir.name}.replaced-{uuid.uuid4().hex}")
        output_dir.rename(backup)
        try:
            temp_dir.rename(output_dir)
        except Exception:
            backup.rename(output_dir)
            raise
        shutil.rmtree(backup)
    else:
        temp_dir.rename(output_dir)


def embed_records(
    selected: list[SongRecord], embedder: Any, config: ChunkConfig, batch_size: int
) -> tuple[list[SongRow], list[tuple[str, str]]]:
    """One L2-normalized mean-of-normalized-chunks vector per song; returns (rows, excluded(id, reason))."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    pending_ids: list[str] = []
    pending_chunks: list[list[int]] = []

    def flush() -> None:
        if not pending_chunks:
            return
        for song_id, vector in zip(pending_ids, embedder.encode_content_batches(pending_chunks, batch_size)):
            if song_id not in sums:
                sums[song_id] = np.zeros(EMBEDDING_DIMENSION, dtype=np.float64)
            sums[song_id] += vector
            counts[song_id] = counts.get(song_id, 0) + 1
        pending_ids.clear()
        pending_chunks.clear()

    for position, record in enumerate(selected, start=1):
        ids = embedder.tokenize(record.lyrics)
        for chunk in chunk_token_ids(ids, config):
            pending_ids.append(record.song_id)
            pending_chunks.append(chunk)
            if len(pending_chunks) >= batch_size:
                flush()
        if position % 250 == 0 or position == len(selected):
            print(f"Prepared/embedded lyrics for {position}/{len(selected)} songs", flush=True)
    flush()

    rows: list[SongRow] = []
    excluded: list[tuple[str, str]] = []
    for record in selected:
        count = counts.get(record.song_id, 0)
        try:
            if count == 0:
                raise EmbeddingError("song produced no chunks")
            vector = l2_normalize((sums[record.song_id] / count).astype(np.float32))
        except EmbeddingError as exc:
            excluded.append((record.song_id, f"invalid vector: {exc}"))
            continue
        metadata = record.metadata()
        row = SongRow(record.song_id, vector, record.artist, record.song, record.link, metadata["excerpt"])
        problems = row_problems(row)
        if problems:
            # Never truncate: over-limit or malformed fields exclude the song, with the reason recorded.
            excluded.append((record.song_id, "; ".join(problems)))
            continue
        rows.append(row)
    return rows, excluded


def write_faiss_backup(rows: list[SongRow], legacy_manifest: dict[str, Any], output_dir: Path, overwrite: bool) -> None:
    """Optional legacy artifact (IndexFlatIP + metadata.json) for baseline comparison; not used at runtime."""
    import faiss

    matrix = np.ascontiguousarray(np.vstack([row.vector for row in rows]), dtype=np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(matrix)
    metadata = [{"song_id": row.song_id, "artist": row.artist, "song": row.song, "source_link": row.link,
                 "safe_source_url": safe_source_url(row.link),
                 "excerpt": row.lyrics_excerpt} for row in rows]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        faiss.write_index(index, str(temp_dir / "songs.faiss"))
        (temp_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = dict(legacy_manifest, indexed_song_count=int(index.ntotal),
                        raw_vector_storage_bytes=int(index.ntotal * EMBEDDING_DIMENSION * 4),
                        faiss_index_file_bytes=(temp_dir / "songs.faiss").stat().st_size)
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _atomic_publish(temp_dir, output_dir, overwrite)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_index(
    csv_path: str | Path,
    collection: str,
    sample_size: int | None = None,
    seed: int = DEFAULT_SAMPLE_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    upsert_batch_size: int = 500,
    faiss_backup: str | Path | None = None,
    overwrite: bool = False,
    settings: MilvusSettings | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    settings = (settings or MilvusSettings.from_env()).with_collection(collection)
    build_mode = "sample" if sample_size is not None else "full"
    check_collection_matches_build_mode(settings.collection, build_mode)
    if faiss_backup is not None and Path(faiss_backup).exists() and not overwrite:
        raise BuildError(f"FAISS backup directory exists: {faiss_backup}. Use --overwrite only after checking it.")
    csv_path = Path(csv_path)
    print(f"Preprocessing {csv_path}…", flush=True)
    records, report = load_and_clean_csv(csv_path)
    selected = sample_records(records, sample_size, seed)
    report.selected_for_build = len(selected)
    print(
        f"Rows: original={report.original_rows}, empty_lyrics={report.missing_or_empty_lyrics}, "
        f"duplicates={report.duplicate_rows_removed}, valid={report.valid_cleaned_count}, selected={len(selected)}",
        flush=True,
    )
    if not selected:
        raise BuildError("No usable lyrics remain after cleaning")

    embedder = SentenceTransformerEmbedder(device=device)
    config = ChunkConfig(chunk_size, overlap)
    config.validate(embedder.info.input_limit, embedder.info.special_tokens)
    chunking = {"content_token_chunk_size": chunk_size, "overlap": overlap, "stride": config.stride}
    model = {"identifier": embedder.info.model_id, "resolved_revision": embedder.info.revision}
    fingerprint = dataset_fingerprint(csv_path)

    def manifest_for(rows_count: int, excluded: list[tuple[str, str]]) -> dict[str, Any]:
        return new_manifest(
            settings.collection, model=model, tokenizer_model_input_limit=embedder.info.input_limit,
            special_tokens_per_input=embedder.info.special_tokens, chunking=chunking, pooling=POOLING_DESCRIPTION,
            preprocessing=report.to_dict(), dataset_fingerprint=fingerprint, build_mode=build_mode,
            sample_seed=seed if sample_size is not None else None, expected_song_count=rows_count, excluded=excluded,
            embeddings="generated from CSV lyrics with the frozen pretrained model",
            source={"type": "csv_build", "csv_path": str(csv_path)}, dependency_versions=dependency_versions(),
        )

    # Fail fast, before the expensive embedding pass, if the destination is unreachable or incompatible.
    client = connect(settings)
    try:
        prepare_destination(client, manifest_for(len(selected), []), settings.timeout, dry_run=True)
        rows, excluded = embed_records(selected, embedder, config, batch_size)
        if not rows:
            raise BuildError("No valid vectors were produced")
        manifest = manifest_for(len(rows), excluded)
        manifest["source"]["embedding_seconds"] = time.perf_counter() - started
        manifest["source"]["process_memory_bytes_after_embedding"] = _memory_bytes()
        def batches():
            for start in range(0, len(rows), upsert_batch_size):
                yield rows[start:start + upsert_batch_size]

        result = import_rows(client, manifest, batches, [row.song_id for row in rows], settings.timeout)
    finally:
        client.close()

    if faiss_backup is not None:
        legacy = {key: manifest[key] for key in ("model", "embedding_dimension", "tokenizer_model_input_limit",
                                                 "special_tokens_per_input", "chunking", "pooling", "preprocessing",
                                                 "dataset_fingerprint", "build_mode", "sample_seed")}
        legacy.update(format_version=1, similarity="cosine similarity via inner product of L2-normalized vectors",
                      excluded_invalid_vectors=len(excluded), build_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      dependency_versions=dependency_versions())
        write_faiss_backup(rows, legacy, Path(faiss_backup), overwrite)
        print(f"Wrote FAISS baseline backup to {faiss_backup}", flush=True)
    duration = time.perf_counter() - started
    print(f"Imported {result.imported_count} songs into Milvus collection {result.collection!r} in {duration:.1f}s "
          f"(excluded {len(excluded)}). Build state: {result.manifest['build_state']}. "
          f"Serve it by setting MILVUS_COLLECTION={result.collection}.", flush=True)
    return result.manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lyrics semantic-search collection in Milvus from the CSV.")
    parser.add_argument("--csv", default=str(default_csv_path()), help="Path to spotify_millsongdata.csv")
    parser.add_argument("--collection", required=True, help="Destination collection: songs_sample or songs_full")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-size", type=int, help=f"Build deterministic sample (normally {DEFAULT_SAMPLE_SIZE})")
    mode.add_argument("--full", action="store_true", help="Explicitly document a full build (the default without --sample-size)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Model inference batch size (chunks)")
    parser.add_argument("--upsert-batch-size", type=int, default=500, help=f"Rows per Milvus upsert (1-{MAX_BATCH_SIZE})")
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cpu or cuda")
    parser.add_argument("--faiss-backup", help="Optional: also write a legacy FAISS artifact directory for baseline checks")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --faiss-backup directory")
    args = parser.parse_args(argv)
    if not 1 <= args.upsert_batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--upsert-batch-size must be from 1 to {MAX_BATCH_SIZE}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file()
    try:
        build_index(args.csv, args.collection, args.sample_size, args.seed, args.chunk_size, args.overlap,
                    args.batch_size, args.device, args.upsert_batch_size, args.faiss_backup, args.overwrite)
    except (ConfigError, RecordValidationError, BuildError) as exc:
        print(f"Build refused: {exc}", file=sys.stderr)
        return 1
    except NotReadyError as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

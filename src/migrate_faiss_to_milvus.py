"""Migrate a completed FAISS IndexFlatIP artifact into a Milvus collection, reusing its stored vectors.

The source files are only read; they stay in place as backups. Re-running the same migration is safe:
IDs are the existing song IDs and every batch is a full-record upsert.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import ConfigError, MilvusSettings, load_env_file
from .faiss_artifacts import ArtifactExportError, FaissArtifact, load_faiss_artifact, validate_artifact_rows, verify_against_csv
from .milvus_store import (
    MAX_BATCH_SIZE,
    IncompatibleCollectionError,
    MilvusUnavailableError,
    NotReadyError,
    RecordValidationError,
    check_collection_matches_build_mode,
    connect,
    dependency_versions,
    import_rows,
    new_manifest,
    prepare_destination,
)

EXIT_INVALID, EXIT_UNAVAILABLE = 1, 3


def manifest_for_artifact(artifact: FaissArtifact, collection: str) -> dict[str, Any]:
    source = artifact.manifest
    excluded = int(source.get("excluded_invalid_vectors") or 0)
    return new_manifest(
        collection,
        model=source["model"],
        tokenizer_model_input_limit=source["tokenizer_model_input_limit"],
        special_tokens_per_input=source["special_tokens_per_input"],
        chunking=source["chunking"],
        pooling=source["pooling"],
        preprocessing=source["preprocessing"],
        dataset_fingerprint=source["dataset_fingerprint"],
        build_mode=source["build_mode"],
        sample_seed=source.get("sample_seed"),
        expected_song_count=artifact.count,
        # The legacy manifest only recorded a count of invalid vectors, not their IDs.
        excluded=[("unknown", "invalid vector excluded by original FAISS build")] * excluded,
        embeddings="reused: exact vectors reconstructed from the FAISS IndexFlatIP artifact (no re-embedding)",
        source={"type": "faiss_artifact", "artifact_dir": str(artifact.directory), "index_file": str(artifact.index_path),
                "faiss_index_type": artifact.index_type, "faiss_index_file_bytes": source.get("faiss_index_file_bytes"),
                "original_build_timestamp_utc": source.get("build_timestamp_utc"),
                "original_build_duration_seconds": source.get("build_duration_seconds"),
                "original_dependency_versions": source.get("dependency_versions")},
        dependency_versions=dependency_versions(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a FAISS IndexFlatIP artifact into Milvus without re-embedding.")
    parser.add_argument("--artifacts", required=True, help="Artifact directory containing songs.faiss, metadata.json, manifest.json")
    parser.add_argument("--faiss-index", help="Override path to songs.faiss")
    parser.add_argument("--metadata", help="Override path to metadata.json")
    parser.add_argument("--manifest", help="Override path to manifest.json")
    parser.add_argument("--collection", required=True, help="Destination collection, e.g. songs_sample or songs_full")
    parser.add_argument("--batch-size", type=int, default=500, help=f"Rows per read/upsert batch (1-{MAX_BATCH_SIZE})")
    parser.add_argument("--csv", help="Optional original CSV: also verify fingerprint and song_id -> song mapping")
    parser.add_argument("--dry-run", action="store_true", help="Validate source and destination; write nothing")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be from 1 to {MAX_BATCH_SIZE}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file()
    try:
        settings = MilvusSettings.from_env(collection=args.collection)
        artifact = load_faiss_artifact(args.artifacts, args.faiss_index, args.metadata, args.manifest)
        print(f"Source: {artifact.index_type} with {artifact.count} vectors (d={artifact.index.d}), "
              f"build_mode={artifact.manifest['build_mode']}, model={artifact.manifest['model']['identifier']} "
              f"@ {artifact.manifest['model'].get('resolved_revision')}", flush=True)
        check_collection_matches_build_mode(settings.collection, artifact.manifest["build_mode"])
        stats = validate_artifact_rows(artifact, args.batch_size)
        print(f"Validated {stats['records']} records: unique IDs, text limits, finite vectors, "
              f"norms {stats['min_vector_norm']:.7f}-{stats['max_vector_norm']:.7f}", flush=True)
        if args.csv:
            print(json.dumps(verify_against_csv(artifact, args.csv)), flush=True)
        manifest = manifest_for_artifact(artifact, settings.collection)
    except (ConfigError, ArtifactExportError, RecordValidationError) as exc:
        print(f"Migration refused: {exc}", file=sys.stderr)
        return EXIT_INVALID

    try:
        client = connect(settings)
    except MilvusUnavailableError as exc:
        prefix = "Dry run: source is valid, but the destination could not be checked. " if args.dry_run else ""
        print(f"{prefix}{exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    try:
        if args.dry_run:
            is_new = prepare_destination(client, manifest, settings.timeout, dry_run=True)
            action = "create it" if is_new else f"upsert {artifact.count} full records (same IDs replace existing ones)"
            print(f"Dry run OK: collection {settings.collection!r} at {settings.safe_uri} is "
                  f"{'absent' if is_new else 'compatible'}; a real run would {action}. Nothing was written.")
            return 0
        result = import_rows(client, manifest, lambda: artifact.iter_rows(args.batch_size), artifact.song_ids,
                             settings.timeout)
    except (IncompatibleCollectionError, RecordValidationError) as exc:
        print(f"Migration refused: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except NotReadyError as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        client.close()
    verification = result.manifest["verification"]
    print(f"Migrated {result.imported_count} songs into {result.collection!r} "
          f"({'new' if result.created_collection else 'existing'} collection); embeddings reused, not regenerated. "
          f"Verified count, ID set, and {verification['records_read_back']} vector/metadata read-backs "
          f"(max |diff| {verification['max_abs_vector_difference']:.2e}). Build state: {result.manifest['build_state']}. "
          f"Source files left unchanged in {artifact.directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

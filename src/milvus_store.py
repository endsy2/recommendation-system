"""Milvus collection schema, validation, build manifests, batched import, and runtime search.

Everything that talks to Milvus goes through a ``pymilvus.MilvusClient`` (or a test double exposing
the same methods). The module never drops, truncates, or recreates an existing collection.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .config import (
    EMBEDDING_DIMENSION,
    FULL_COLLECTION,
    MANIFEST_COLLECTION,
    SAMPLE_COLLECTION,
    MilvusSettings,
    validate_collection_name,
)

SCHEMA_VERSION = 1
MANIFEST_FORMAT_VERSION = 1
PK_FIELD = "song_id"
VECTOR_FIELD = "vector"
INDEX_NAME = "vector_hnsw_cosine"
INDEX_TYPE = "HNSW"
METRIC_TYPE = "COSINE"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_SEARCH_EF = 64
INDEX_BUILD_PARAMS = {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}
SEARCH_PARAMS = {"metric_type": METRIC_TYPE, "params": {"ef": HNSW_SEARCH_EF}}
# VARCHAR max_length is in UTF-8 bytes. Observed maxima over all 57,649 cleaned songs: song_id 5,
# artist 44, song 77, link 102, excerpt 303. Limits leave headroom; nothing is ever truncated.
FIELD_BYTE_LIMITS = {PK_FIELD: 64, "artist": 512, "song": 512, "link": 2048, "lyrics_excerpt": 2048}
NULLABLE_FIELDS = frozenset({"link", "lyrics_excerpt"})
OUTPUT_FIELDS = ["artist", "song", "link", "lyrics_excerpt"]
NORM_TOLERANCE = 1e-4
BUILD_IMPORTING = "importing"
BUILD_COMPLETE = "complete"
MAX_BATCH_SIZE = 5000
_REGISTRY_VECTOR = [1.0, 0.0]
_REGISTRY_JSON_LIMIT = 65535
_EXCLUDED_EXAMPLE_LIMIT = 100
START_MILVUS_HINT = (
    "Start the local server with: docker compose -f deploy/milvus/docker-compose.yml up -d "
    "(see README 'Start Milvus'), then check MILVUS_URI/MILVUS_TOKEN."
)


class NotReadyError(RuntimeError):
    """The configured search backend cannot serve requests. Messages are safe to show to users."""


class MilvusUnavailableError(NotReadyError):
    pass


class CollectionNotReadyError(NotReadyError):
    pass


class IncompatibleCollectionError(NotReadyError):
    pass


class RecordValidationError(ValueError):
    pass


# --------------------------------------------------------------------------- connection

def connect(settings: MilvusSettings, client_factory: Callable[..., Any] | None = None) -> Any:
    """Open a MilvusClient against a running server. Never falls back to Milvus Lite or FAISS."""
    if client_factory is None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise MilvusUnavailableError("pymilvus is not installed; run: python -m pip install -r requirements.txt") from exc
        client_factory = MilvusClient
    try:
        return client_factory(uri=settings.uri, token=settings.token, timeout=settings.timeout)
    except Exception as exc:
        raise MilvusUnavailableError(
            f"Could not connect to Milvus at {settings.safe_uri} within {settings.timeout:g}s "
            f"({type(exc).__name__}). {START_MILVUS_HINT}"
        ) from exc


def _call(action: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one Milvus RPC, turning SDK/transport failures into a user-safe NotReadyError."""
    try:
        return func(*args, **kwargs)
    except NotReadyError:
        raise
    except Exception as exc:
        raise MilvusUnavailableError(f"Milvus request failed while trying to {action} ({type(exc).__name__}). "
                                     f"{START_MILVUS_HINT}") from exc


# --------------------------------------------------------------------------- schema and index

def build_schema() -> Any:
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=False,
        description=f"Lyrics-only song vectors; song-search schema v{SCHEMA_VERSION}",
    )
    schema.add_field(field_name=PK_FIELD, datatype=DataType.VARCHAR, is_primary=True, auto_id=False,
                     max_length=FIELD_BYTE_LIMITS[PK_FIELD])
    schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
    schema.add_field(field_name="artist", datatype=DataType.VARCHAR, max_length=FIELD_BYTE_LIMITS["artist"])
    schema.add_field(field_name="song", datatype=DataType.VARCHAR, max_length=FIELD_BYTE_LIMITS["song"])
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=FIELD_BYTE_LIMITS["link"], nullable=True)
    schema.add_field(field_name="lyrics_excerpt", datatype=DataType.VARCHAR,
                     max_length=FIELD_BYTE_LIMITS["lyrics_excerpt"], nullable=True)
    return schema


def build_index_params(client: Any, params: Mapping[str, Any] | None = None) -> Any:
    # Explicit HNSW + COSINE: graph index with M=16, efConstruction=200.
    index_params = client.prepare_index_params()
    build_params = dict(INDEX_BUILD_PARAMS if params is None else params)
    index_params.add_index(field_name=VECTOR_FIELD, index_type=INDEX_TYPE, index_name=INDEX_NAME,
                           metric_type=METRIC_TYPE, params=build_params)
    return index_params


_EXPECTED_FIELDS: dict[str, dict[str, Any]] = {
    PK_FIELD: {"type": "VARCHAR", "max_length": FIELD_BYTE_LIMITS[PK_FIELD], "is_primary": True},
    VECTOR_FIELD: {"type": "FLOAT_VECTOR", "dim": EMBEDDING_DIMENSION},
    "artist": {"type": "VARCHAR", "max_length": FIELD_BYTE_LIMITS["artist"]},
    "song": {"type": "VARCHAR", "max_length": FIELD_BYTE_LIMITS["song"]},
    "link": {"type": "VARCHAR", "max_length": FIELD_BYTE_LIMITS["link"]},
    "lyrics_excerpt": {"type": "VARCHAR", "max_length": FIELD_BYTE_LIMITS["lyrics_excerpt"]},
}


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def schema_problems(description: Mapping[str, Any]) -> list[str]:
    """Compare a describe_collection() result with the exact v1 schema. Empty list means compatible."""
    problems: list[str] = []
    if description.get("enable_dynamic_field"):
        problems.append("dynamic fields are enabled (expected disabled)")
    if description.get("auto_id"):
        problems.append("automatic primary-key generation is enabled (expected auto_id=False)")
    if description.get("functions"):
        problems.append("collection defines functions; expected plain stored vectors")
    fields = {str(field.get("name")): field for field in description.get("fields", [])}
    missing = sorted(set(_EXPECTED_FIELDS) - set(fields))
    extra = sorted(set(fields) - set(_EXPECTED_FIELDS))
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected fields: {', '.join(extra)}")
    for name, expected in _EXPECTED_FIELDS.items():
        field = fields.get(name)
        if field is None:
            continue
        actual_type = _type_name(field.get("type"))
        if actual_type != expected["type"]:
            problems.append(f"field {name} has type {actual_type}, expected {expected['type']}")
        params = field.get("params") or {}
        for key in ("dim", "max_length"):
            if key in expected:
                try:
                    actual = int(params.get(key))
                except (TypeError, ValueError):
                    actual = None
                if actual != expected[key]:
                    problems.append(f"field {name} has {key}={params.get(key)}, expected {expected[key]}")
        if bool(field.get("is_primary")) != bool(expected.get("is_primary")):
            problems.append(f"field {name} primary-key flag is {bool(field.get('is_primary'))}, "
                            f"expected {bool(expected.get('is_primary'))}")
        if field.get("auto_id"):
            problems.append(f"field {name} has auto_id enabled")
        if bool(field.get("nullable")) != (name in NULLABLE_FIELDS):
            problems.append(f"field {name} nullable={bool(field.get('nullable'))}, expected {name in NULLABLE_FIELDS}")
    return problems


def index_problems(indexes: Sequence[Mapping[str, Any]]) -> list[str]:
    vector_indexes = [index for index in indexes if index.get("field_name") == VECTOR_FIELD]
    if not vector_indexes:
        return [f"no index on field {VECTOR_FIELD}; expected {INDEX_TYPE} with {METRIC_TYPE}"]
    if len(vector_indexes) > 1:
        return [f"multiple indexes on field {VECTOR_FIELD}"]
    index = vector_indexes[0]
    problems = []
    if str(index.get("index_type", "")).upper() != INDEX_TYPE:
        problems.append(f"vector index type is {index.get('index_type')}, expected {INDEX_TYPE}")
    if str(index.get("metric_type", "")).upper() != METRIC_TYPE:
        problems.append(f"vector metric is {index.get('metric_type')}, expected {METRIC_TYPE}")
    return problems


def describe_indexes(client: Any, collection: str, timeout: float) -> list[dict[str, Any]]:
    names = _call("list indexes", client.list_indexes, collection)
    result = []
    for name in names:
        info = _call("describe an index", client.describe_index, collection, name, timeout=timeout)
        if info:
            result.append(dict(info))
    return result


def validate_song_collection(client: Any, collection: str, timeout: float) -> None:
    description = _call("describe the collection", client.describe_collection, collection, timeout=timeout)
    problems = schema_problems(description) + index_problems(describe_indexes(client, collection, timeout))
    if problems:
        raise IncompatibleCollectionError(
            f"Collection {collection!r} is not compatible with song-search schema v{SCHEMA_VERSION}: "
            + "; ".join(problems) + ". It was not modified. Use a new collection name, or drop it manually "
            "after checking its contents."
        )


def load_state_name(state: Any) -> str:
    if isinstance(state, Mapping):
        state = state.get("state")
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    text = str(state)
    if text.startswith("<") and ":" in text:
        text = text.split(":", 1)[1].strip(" >")
    return text


def ensure_loaded(client: Any, collection: str, timeout: float, allow_load: bool = True) -> None:
    """Make sure the collection is loaded (searchable). Loading is non-destructive."""
    state = load_state_name(_call("check the load state", client.get_load_state, collection, timeout=timeout))
    if state == "Loaded":
        return
    if state == "NotExist":
        raise CollectionNotReadyError(f"Collection {collection!r} does not exist.")
    if not allow_load:
        raise CollectionNotReadyError(f"Collection {collection!r} is not loaded (state: {state}).")
    if state != "Loading":
        _call("load the collection", client.load_collection, collection, timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        state = load_state_name(_call("check the load state", client.get_load_state, collection, timeout=timeout))
        if state == "Loaded":
            return
        if time.monotonic() >= deadline:
            raise CollectionNotReadyError(f"Collection {collection!r} did not finish loading within {timeout:g}s "
                                          f"(state: {state}).")
        time.sleep(0.2)


# --------------------------------------------------------------------------- build manifest registry

def _registry_schema() -> Any:
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                        description="Song-search build manifests, one row per song collection")
    schema.add_field(field_name="collection_name", datatype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255)
    schema.add_field(field_name="build_state", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="updated_at_utc", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="manifest_json", datatype=DataType.VARCHAR, max_length=_REGISTRY_JSON_LIMIT)
    # Milvus collections need a vector field; this constant 2-d placeholder is never searched.
    schema.add_field(field_name="registry_vector", datatype=DataType.FLOAT_VECTOR, dim=len(_REGISTRY_VECTOR))
    return schema


_REGISTRY_FIELDS = {"collection_name": "VARCHAR", "build_state": "VARCHAR", "updated_at_utc": "VARCHAR",
                    "manifest_json": "VARCHAR", "registry_vector": "FLOAT_VECTOR"}


def ensure_registry(client: Any, timeout: float) -> None:
    if not _call("check the manifest registry", client.has_collection, MANIFEST_COLLECTION, timeout=timeout):
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="registry_vector", index_type="FLAT", index_name="registry_vector_flat",
                               metric_type="COSINE", params={})
        _call("create the manifest registry", client.create_collection, MANIFEST_COLLECTION,
              schema=_registry_schema(), index_params=index_params, timeout=timeout)
    _validate_registry(client, timeout)
    ensure_loaded(client, MANIFEST_COLLECTION, timeout)


def _validate_registry(client: Any, timeout: float) -> None:
    description = _call("describe the manifest registry", client.describe_collection, MANIFEST_COLLECTION, timeout=timeout)
    fields = {str(field.get("name")): _type_name(field.get("type")) for field in description.get("fields", [])}
    if fields != _REGISTRY_FIELDS:
        raise IncompatibleCollectionError(
            f"Collection {MANIFEST_COLLECTION!r} exists but is not a song-search manifest registry "
            f"(fields: {sorted(fields)}). It was not modified."
        )


def read_manifest(client: Any, collection: str, timeout: float) -> dict[str, Any] | None:
    """Return the manifest stored for ``collection``, or None if the registry has no entry."""
    if not _call("check the manifest registry", client.has_collection, MANIFEST_COLLECTION, timeout=timeout):
        return None
    _validate_registry(client, timeout)
    ensure_loaded(client, MANIFEST_COLLECTION, timeout)
    rows = _call("read the build manifest", client.get, MANIFEST_COLLECTION, ids=[collection],
                 output_fields=["collection_name", "build_state", "manifest_json"], timeout=timeout,
                 consistency_level="Strong")
    if not rows:
        return None
    row = rows[0]
    try:
        manifest = json.loads(row["manifest_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompatibleCollectionError(f"Build manifest for {collection!r} is unreadable.") from exc
    if not isinstance(manifest, dict) or manifest.get("build_state") != row.get("build_state"):
        raise IncompatibleCollectionError(f"Build manifest for {collection!r} is inconsistent.")
    return manifest


def write_manifest(client: Any, manifest: Mapping[str, Any], timeout: float) -> None:
    collection = manifest["collection"]["name"]
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    if len(text.encode("utf-8")) > _REGISTRY_JSON_LIMIT:
        raise RecordValidationError("Build manifest is larger than the 65,535-byte registry limit")
    row = {"collection_name": collection, "build_state": str(manifest["build_state"]),
           "updated_at_utc": utc_now(), "manifest_json": text, "registry_vector": list(_REGISTRY_VECTOR)}
    result = _call("write the build manifest", client.upsert, MANIFEST_COLLECTION, data=[row], timeout=timeout)
    if int(_upsert_count(result)) != 1:
        raise MilvusUnavailableError(f"Milvus did not confirm the manifest write for {collection!r}")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dependency_versions() -> dict[str, str | None]:
    import importlib.metadata

    result: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "sentence-transformers", "faiss-cpu", "pymilvus", "fastapi", "pydantic"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def content_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must match before new records may be written into an existing collection."""
    model = manifest.get("model") or {}
    preprocessing = manifest.get("preprocessing") or {}
    return {
        "schema_version": (manifest.get("collection") or {}).get("schema_version"),
        "model_identifier": model.get("identifier"),
        "model_revision": model.get("resolved_revision"),
        "embedding_dimension": manifest.get("embedding_dimension"),
        "tokenizer_model_input_limit": manifest.get("tokenizer_model_input_limit"),
        "special_tokens_per_input": manifest.get("special_tokens_per_input"),
        "chunking": manifest.get("chunking"),
        "pooling": manifest.get("pooling"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "build_mode": manifest.get("build_mode"),
        "sample_seed": manifest.get("sample_seed"),
        "selected_for_build": preprocessing.get("selected_for_build"),
    }


def signature_differences(existing: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    old_sig, new_sig = content_signature(existing), content_signature(new)
    return [f"{key}: existing={old_sig[key]!r}, new={new_sig[key]!r}" for key in new_sig if old_sig[key] != new_sig[key]]


def check_collection_matches_build_mode(collection: str, build_mode: str) -> None:
    """Keep the 1,000-song development sample and the full catalogue in separate collections."""
    if build_mode not in ("sample", "full"):
        raise RecordValidationError(f"Unknown build mode {build_mode!r}")
    lowered = collection.lower()
    if build_mode == "sample" and (collection == FULL_COLLECTION or "full" in lowered):
        raise RecordValidationError(f"Refusing to write a sample build into {collection!r}; use {SAMPLE_COLLECTION!r} "
                                    "or another name without 'full'.")
    if build_mode == "full" and (collection == SAMPLE_COLLECTION or "sample" in lowered):
        raise RecordValidationError(f"Refusing to write a full build into {collection!r}; use {FULL_COLLECTION!r} "
                                    "or another name without 'sample'.")


def new_manifest(
    collection: str,
    *,
    model: Mapping[str, Any],
    tokenizer_model_input_limit: int,
    special_tokens_per_input: int,
    chunking: Mapping[str, Any],
    pooling: str,
    preprocessing: Mapping[str, Any],
    dataset_fingerprint: str,
    build_mode: str,
    sample_seed: int | None,
    expected_song_count: int,
    excluded: Sequence[tuple[str, str]],
    embeddings: str,
    source: Mapping[str, Any],
    dependency_versions: Mapping[str, Any],
) -> dict[str, Any]:
    from .config import NORMALIZATION_DESCRIPTION

    return {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "collection": {"name": collection, "schema_version": SCHEMA_VERSION,
                       "primary_key": f"{PK_FIELD} VARCHAR({FIELD_BYTE_LIMITS[PK_FIELD]}), auto_id=False",
                       "vector_field": VECTOR_FIELD, "index_name": INDEX_NAME, "index_type": INDEX_TYPE,
                       "metric_type": METRIC_TYPE, "index_params": dict(INDEX_BUILD_PARAMS),
                       "field_byte_limits": dict(FIELD_BYTE_LIMITS)},
        "build_state": BUILD_IMPORTING,
        "model": dict(model),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "tokenizer_model_input_limit": tokenizer_model_input_limit,
        "special_tokens_per_input": special_tokens_per_input,
        "chunking": dict(chunking),
        "pooling": pooling,
        "normalization": NORMALIZATION_DESCRIPTION,
        "similarity": "cosine similarity (Milvus COSINE on L2-normalized vectors)",
        "preprocessing": dict(preprocessing),
        "dataset_fingerprint": dataset_fingerprint,
        "build_mode": build_mode,
        "sample_seed": sample_seed,
        "expected_song_count": expected_song_count,
        "imported_song_count": None,
        "excluded_records": summarize_exclusions(excluded),
        "embeddings": embeddings,
        "source": dict(source),
        "dependency_versions": dict(dependency_versions),
    }


def summarize_exclusions(excluded: Sequence[tuple[str, str]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    for _, reason in excluded:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"count": len(excluded), "by_reason": by_reason,
            "examples": [{"song_id": song_id, "reason": reason} for song_id, reason in excluded[:_EXCLUDED_EXAMPLE_LIMIT]]}


# --------------------------------------------------------------------------- records

@dataclass(frozen=True)
class SongRow:
    song_id: str
    vector: np.ndarray
    artist: str
    song: str
    link: str | None
    lyrics_excerpt: str | None

    def entity(self) -> dict[str, Any]:
        return {PK_FIELD: self.song_id, VECTOR_FIELD: np.asarray(self.vector, dtype=np.float32).tolist(),
                "artist": self.artist, "song": self.song, "link": self.link, "lyrics_excerpt": self.lyrics_excerpt}


def row_from_metadata(item: Mapping[str, Any], vector: np.ndarray) -> SongRow:
    """Pair a vector with legacy metadata.json / SongRecord.metadata() fields (excerpt -> lyrics_excerpt)."""
    return SongRow(str(item["song_id"]), np.asarray(vector, dtype=np.float32), item.get("artist"), item.get("song"),
                   item.get("source_link"), item.get("excerpt"))


def _text_problem(name: str, value: Any, nullable: bool) -> str | None:
    if value is None:
        return None if nullable else f"{name} is missing"
    if not isinstance(value, str):
        return f"{name} is not text"
    if not nullable and not value.strip():
        return f"{name} is empty"
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return f"{name} is not valid UTF-8 text"
    if size > FIELD_BYTE_LIMITS[name]:
        return f"{name} is {size} UTF-8 bytes, over the {FIELD_BYTE_LIMITS[name]}-byte limit (not truncated)"
    return None


def vector_problem(vector: Any) -> str | None:
    value = np.asarray(vector)
    if value.shape != (EMBEDDING_DIMENSION,):
        return f"vector has shape {value.shape}, expected ({EMBEDDING_DIMENSION},)"
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        return "vector contains non-finite values"
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not math.isclose(norm, 1.0, abs_tol=NORM_TOLERANCE):
        return f"vector is not L2-normalized (norm {norm:.6f})"
    return None


def row_problems(row: SongRow) -> list[str]:
    problems = [problem for problem in (
        _text_problem(PK_FIELD, row.song_id, False),
        _text_problem("artist", row.artist, False),
        _text_problem("song", row.song, False),
        _text_problem("link", row.link, True),
        _text_problem("lyrics_excerpt", row.lyrics_excerpt, True),
        vector_problem(row.vector),
    ) if problem]
    return problems


def duplicate_ids(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for song_id in ids:
        (duplicates if song_id in seen else seen).add(song_id)
    return sorted(duplicates)


def _upsert_count(result: Any) -> int:
    if isinstance(result, Mapping):
        return int(result.get("upsert_count", 0))
    return int(getattr(result, "upsert_count", 0))


# --------------------------------------------------------------------------- import

@dataclass
class ImportResult:
    collection: str
    imported_count: int
    manifest: dict[str, Any]
    created_collection: bool


def prepare_destination(client: Any, manifest: dict[str, Any], timeout: float, dry_run: bool = False) -> bool:
    """Validate (and unless dry_run, create) the destination. Returns True if the collection is new.

    An existing collection must have the exact schema/index and a manifest with the same content
    signature; otherwise nothing is written. Existing collections are never dropped or recreated.
    """
    collection = manifest["collection"]["name"]
    validate_collection_name(collection)
    check_collection_matches_build_mode(collection, manifest["build_mode"])
    exists = _call("check the collection", client.has_collection, collection, timeout=timeout)
    if exists:
        validate_song_collection(client, collection, timeout)
        existing = read_manifest(client, collection, timeout)
        if existing is None:
            raise IncompatibleCollectionError(
                f"Collection {collection!r} exists but has no song-search build manifest, so its contents are "
                "unknown. It was not modified; choose a new collection name."
            )
        differences = signature_differences(existing, manifest)
        if differences:
            raise IncompatibleCollectionError(
                f"Collection {collection!r} holds a different dataset/embedding build ({'; '.join(differences)}). "
                "It was not modified. Use a new collection for incompatible rebuilds."
            )
        return False
    if dry_run:
        if _call("check the manifest registry", client.has_collection, MANIFEST_COLLECTION, timeout=timeout):
            _validate_registry(client, timeout)
        return True
    ensure_registry(client, timeout)
    # Record the in-progress state before the collection exists, so a crash never leaves an
    # unexplained collection behind.
    write_manifest(client, manifest, timeout)
    _call("create the collection", client.create_collection, collection, schema=build_schema(),
          index_params=build_index_params(client), consistency_level="Bounded", timeout=timeout)
    validate_song_collection(client, collection, timeout)
    return True


def import_rows(
    client: Any,
    manifest: dict[str, Any],
    batches: Callable[[], Iterable[Sequence[SongRow]]],
    expected_ids: Sequence[str],
    timeout: float,
) -> ImportResult:
    """Batched full-record upserts, then verification, then promotion of the manifest to complete.

    ``batches`` is called twice (write, then verify) and must yield the same validated rows each time.
    Re-running with the same rows is idempotent: upsert replaces entities that share a primary key.
    """
    collection = manifest["collection"]["name"]
    duplicates = duplicate_ids(expected_ids)
    if duplicates:
        raise RecordValidationError(f"Duplicate source song IDs: {', '.join(duplicates[:20])}")
    if not expected_ids:
        raise RecordValidationError("No records to import")
    manifest = dict(manifest, build_state=BUILD_IMPORTING, started_at_utc=utc_now())
    created = prepare_destination(client, manifest, timeout)
    ensure_registry(client, timeout)
    write_manifest(client, manifest, timeout)
    ensure_loaded(client, collection, timeout)

    written = 0
    for batch in batches():
        if not batch:
            continue
        entities = [row.entity() for row in batch]
        result = _call("upsert songs", client.upsert, collection, data=entities, timeout=timeout)
        if _upsert_count(result) != len(entities):
            raise MilvusUnavailableError(f"Milvus confirmed {_upsert_count(result)} of {len(entities)} upserted rows")
        written += len(entities)
        print(f"Upserted {written}/{len(expected_ids)} songs into {collection}", flush=True)
    if written != len(expected_ids):
        raise RecordValidationError(f"Wrote {written} rows but expected {len(expected_ids)}")

    verification = verify_collection_contents(client, collection, batches, expected_ids, timeout)
    manifest = dict(manifest, build_state=BUILD_COMPLETE, imported_song_count=verification["searchable_count"],
                    verification=verification, completed_at_utc=utc_now())
    write_manifest(client, manifest, timeout)
    return ImportResult(collection, verification["searchable_count"], manifest, created)


def count_entities(client: Any, collection: str, timeout: float) -> int:
    rows = _call("count songs", client.query, collection, filter="", output_fields=["count(*)"],
                 timeout=timeout, consistency_level="Strong")
    return int(rows[0]["count(*)"]) if rows else 0


def iter_ids(client: Any, collection: str, timeout: float, batch_size: int = 1000) -> Iterator[str]:
    iterator = _call("list song IDs", client.query_iterator, collection, batch_size=batch_size, filter="",
                     output_fields=[PK_FIELD], timeout=timeout, consistency_level="Strong")
    try:
        while True:
            page = _call("list song IDs", iterator.next)
            if not page:
                return
            for row in page:
                yield str(row[PK_FIELD])
    finally:
        iterator.close()


def verify_collection_contents(
    client: Any,
    collection: str,
    batches: Callable[[], Iterable[Sequence[SongRow]]],
    expected_ids: Sequence[str],
    timeout: float,
) -> dict[str, Any]:
    """Strong-consistency checks: logical count, exact ID set, and every vector/metadata mapping."""
    searchable = count_entities(client, collection, timeout)
    stored_ids = set(iter_ids(client, collection, timeout))
    expected = set(expected_ids)
    missing, stale = sorted(expected - stored_ids), sorted(stored_ids - expected)
    if missing or stale or searchable != len(expected):
        raise RecordValidationError(
            f"Collection {collection!r} verification failed: count={searchable}, expected={len(expected)}, "
            f"missing={missing[:10]}, unexpected={stale[:10]}. The build stays marked '{BUILD_IMPORTING}'."
        )
    checked, max_diff = 0, 0.0
    for batch in batches():
        if not batch:
            continue
        stored = _call("read back songs", client.get, collection, ids=[row.song_id for row in batch],
                       output_fields=[PK_FIELD, VECTOR_FIELD, *OUTPUT_FIELDS], timeout=timeout,
                       consistency_level="Strong")
        by_id = {str(item[PK_FIELD]): item for item in stored}
        for row in batch:
            item = by_id.get(row.song_id)
            if item is None:
                raise RecordValidationError(f"Song {row.song_id} is not readable after import")
            diff = float(np.max(np.abs(np.asarray(item[VECTOR_FIELD], dtype=np.float32) - row.vector)))
            if diff > 1e-6:
                raise RecordValidationError(f"Stored vector for song {row.song_id} differs from source (max diff {diff})")
            for field_name, value in (("artist", row.artist), ("song", row.song), ("link", row.link),
                                      ("lyrics_excerpt", row.lyrics_excerpt)):
                if item.get(field_name) != value:
                    raise RecordValidationError(f"Stored {field_name} for song {row.song_id} differs from source")
            max_diff = max(max_diff, diff)
            checked += 1
    if checked != len(expected):
        raise RecordValidationError(f"Verified {checked} songs but expected {len(expected)}")
    return {"searchable_count": searchable, "id_set_matches": True, "records_read_back": checked,
            "max_abs_vector_difference": max_diff, "consistency_level": "Strong", "verified_at_utc": utc_now()}


# --------------------------------------------------------------------------- runtime search

@dataclass(frozen=True)
class StoredHit:
    song_id: str
    score: float
    artist: str
    song: str
    link: str | None
    lyrics_excerpt: str | None


def _hit_value(hit: Any, key: str) -> Any:
    getter = getattr(hit, "get", None)
    if callable(getter):
        value = getter(key)
        if value is not None:
            return value
        entity = getter("entity")
        if isinstance(entity, Mapping):
            return entity.get(key)
    return None


def parse_hits(raw_hits: Iterable[Any]) -> list[StoredHit]:
    """Map SDK hits (pk field or 'id'; score under 'distance' or 'score') to plain records."""
    hits: list[StoredHit] = []
    for hit in raw_hits:
        song_id = _hit_value(hit, PK_FIELD)
        if song_id is None:
            song_id = _hit_value(hit, "id")
        score = _hit_value(hit, "distance")
        if score is None:
            score = _hit_value(hit, "score")
        if song_id is None or score is None or not math.isfinite(float(score)):
            raise CollectionNotReadyError("Milvus returned a malformed search hit")
        hits.append(StoredHit(str(song_id), float(score), str(_hit_value(hit, "artist")), str(_hit_value(hit, "song")),
                              _hit_value(hit, "link"), _hit_value(hit, "lyrics_excerpt")))
    # COSINE: larger is more similar. Milvus already returns descending order; a stable sort keeps
    # its order among exact ties and guards against any client-side reordering.
    return sorted(hits, key=lambda item: -item.score)


def validate_serving_manifest(collection: str, manifest: Mapping[str, Any] | None) -> None:
    if manifest is None:
        raise CollectionNotReadyError(
            f"Collection {collection!r} has no build manifest. Migrate or build it first "
            "(python -m src.migrate_faiss_to_milvus ... or python -m src.build_index ...)."
        )
    if manifest.get("manifest_format_version") != MANIFEST_FORMAT_VERSION:
        raise IncompatibleCollectionError(f"Build manifest for {collection!r} has an unsupported format version")
    info = manifest.get("collection") or {}
    if (info.get("name") != collection or info.get("schema_version") != SCHEMA_VERSION
            or info.get("index_type") != INDEX_TYPE or info.get("metric_type") != METRIC_TYPE):
        raise IncompatibleCollectionError(f"Build manifest for {collection!r} does not describe a v{SCHEMA_VERSION} "
                                          f"{INDEX_TYPE}/{METRIC_TYPE} song collection")
    if manifest.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise IncompatibleCollectionError(f"Build manifest for {collection!r} has an incompatible embedding dimension")
    if manifest.get("build_state") != BUILD_COMPLETE:
        raise CollectionNotReadyError(
            f"Collection {collection!r} build is '{manifest.get('build_state')}', not '{BUILD_COMPLETE}'. "
            "It is not served until an import finishes and passes verification."
        )


class MilvusSongStore:
    """Shared, long-lived connection to one validated, completed, loaded song collection."""

    def __init__(self, client: Any, collection: str, timeout: float, manifest: dict[str, Any]) -> None:
        self.client, self.collection, self.timeout, self.manifest = client, collection, timeout, manifest

    @classmethod
    def open(cls, settings: MilvusSettings, client_factory: Callable[..., Any] | None = None) -> "MilvusSongStore":
        client = connect(settings, client_factory)
        try:
            store = cls(client, settings.collection, settings.timeout, {})
            store.manifest = store.check_ready(allow_load=True)
            return store
        except BaseException:
            _close_quietly(client)
            raise

    def check_ready(self, allow_load: bool = False) -> dict[str, Any]:
        """Connectivity, existence, schema/index, completed manifest, load state, and searchable count."""
        if not _call("check the collection", self.client.has_collection, self.collection, timeout=self.timeout):
            raise CollectionNotReadyError(
                f"Collection {self.collection!r} does not exist in Milvus. Migrate or build it, or set "
                "MILVUS_COLLECTION to an existing completed collection."
            )
        validate_song_collection(self.client, self.collection, self.timeout)
        manifest = read_manifest(self.client, self.collection, self.timeout)
        validate_serving_manifest(self.collection, manifest)
        ensure_loaded(self.client, self.collection, self.timeout, allow_load=allow_load)
        count = count_entities(self.client, self.collection, self.timeout)
        if count != manifest.get("imported_song_count"):
            raise CollectionNotReadyError(
                f"Collection {self.collection!r} holds {count} songs but its completed build recorded "
                f"{manifest.get('imported_song_count')}. Re-run verification before serving."
            )
        return manifest

    @property
    def song_count(self) -> int:
        return int(self.manifest.get("imported_song_count", 0))

    def search(
        self,
        vector: np.ndarray,
        limit: int,
        consistency_level: str | None = None,
        search_params: Mapping[str, Any] | None = None,
    ) -> list[StoredHit]:
        extra = {"consistency_level": consistency_level} if consistency_level else {}
        params = dict(SEARCH_PARAMS if search_params is None else search_params)
        result = _call("search songs", self.client.search, self.collection, data=[np.asarray(vector, np.float32).tolist()],
                       anns_field=VECTOR_FIELD, limit=limit,
                       search_params=params,
                       output_fields=list(OUTPUT_FIELDS), timeout=self.timeout, **extra)
        return parse_hits(result[0] if result else [])

    def iter_metadata(self, batch_size: int = 1000) -> Iterator[dict[str, Any]]:
        """All stored song metadata (no vectors), e.g. to build the TF-IDF comparator on the same songs."""
        iterator = _call("list songs", self.client.query_iterator, self.collection, batch_size=batch_size, filter="",
                         output_fields=[PK_FIELD, *OUTPUT_FIELDS], timeout=self.timeout, consistency_level="Strong")
        try:
            while True:
                page = _call("list songs", iterator.next)
                if not page:
                    return
                yield from (dict(row) for row in page)
        finally:
            iterator.close()

    def close(self) -> None:
        _close_quietly(self.client)


def _close_quietly(client: Any) -> None:
    try:
        client.close()
    except Exception:
        pass

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_BATCH_SIZE = 64
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SAMPLE_SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent
DISCOVERED_DATASET_PATH = ROOT_DIR / "dataset" / "archive" / "spotify_millsongdata.csv"
DEFAULT_ARTIFACTS_PATH = ROOT_DIR / "artifacts" / "sample"
REQUIRED_COLUMNS = ("artist", "song", "link", "text")
POOLING_DESCRIPTION = "L2-normalize each chunk; coordinate-wise mean per song; L2-normalize mean"
NORMALIZATION_DESCRIPTION = "chunk, pooled song, and query vectors are all L2-normalized float32"

# Milvus runtime configuration (environment variables; see .env.example).
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_COLLECTION = "songs_sample"
DEFAULT_MILVUS_TIMEOUT = 10.0
MAX_MILVUS_TIMEOUT = 300.0
SAMPLE_COLLECTION = "songs_sample"
FULL_COLLECTION = "songs_full"
# Registry holding one build manifest per song collection; it is never served as a song collection.
MANIFEST_COLLECTION = "song_search_builds"
_COLLECTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
_ALLOWED_URI_SCHEMES = ("http", "https", "tcp")


def default_csv_path() -> Path:
    """Return the project dataset location; callers still validate it exists."""
    return DISCOVERED_DATASET_PATH


class ConfigError(ValueError):
    """Raised when Milvus settings are missing or invalid."""


def load_env_file(path: Path | None = None) -> None:
    """Load a project .env file if present. Real environment variables always win."""
    env_path = path or ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def validate_collection_name(name: str) -> str:
    if not isinstance(name, str) or not _COLLECTION_NAME.match(name):
        raise ConfigError(
            f"Invalid Milvus collection name {name!r}: use 1-255 letters, digits, or underscores, "
            "starting with a letter or underscore."
        )
    if name == MANIFEST_COLLECTION:
        raise ConfigError(f"{MANIFEST_COLLECTION!r} is the build-manifest registry, not a song collection.")
    return name


def validate_uri(uri: str) -> str:
    value = (uri or "").strip()
    if not value:
        raise ConfigError(f"MILVUS_URI is empty. For local Milvus Standalone use {DEFAULT_MILVUS_URI}.")
    parts = urlsplit(value)
    if parts.scheme.lower() not in _ALLOWED_URI_SCHEMES or not parts.hostname:
        hint = ""
        if value.endswith(".db") or parts.scheme in ("", "file"):
            hint = " Milvus Lite database files are intentionally not supported; start Milvus Standalone instead."
        raise ConfigError(
            f"MILVUS_URI must be an http(s):// or tcp:// server address such as {DEFAULT_MILVUS_URI}.{hint}"
        )
    try:
        parts.port
    except ValueError as exc:
        raise ConfigError("MILVUS_URI has an invalid port") from exc
    return value


def redact_uri(uri: str) -> str:
    """Remove user:password from a URI before it appears in logs or error messages."""
    parts = urlsplit(uri)
    if parts.username is None and parts.password is None:
        return uri
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


@dataclass(frozen=True)
class MilvusSettings:
    uri: str = DEFAULT_MILVUS_URI
    token: str = field(default="", repr=False)
    collection: str = DEFAULT_MILVUS_COLLECTION
    timeout: float = DEFAULT_MILVUS_TIMEOUT

    def __post_init__(self) -> None:
        validate_uri(self.uri)
        validate_collection_name(self.collection)
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ConfigError("MILVUS_TIMEOUT must be a number of seconds")
        if not 0 < float(self.timeout) <= MAX_MILVUS_TIMEOUT:
            raise ConfigError(f"MILVUS_TIMEOUT must be greater than 0 and at most {MAX_MILVUS_TIMEOUT:g} seconds")

    def __repr__(self) -> str:
        return (f"MilvusSettings(uri={self.safe_uri!r}, token={'***' if self.token else ''!r}, "
                f"collection={self.collection!r}, timeout={self.timeout!r})")

    @property
    def safe_uri(self) -> str:
        return redact_uri(self.uri)

    def with_collection(self, collection: str) -> "MilvusSettings":
        return MilvusSettings(self.uri, self.token, collection, self.timeout)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, collection: str | None = None) -> "MilvusSettings":
        env = os.environ if environ is None else environ
        raw_timeout = (env.get("MILVUS_TIMEOUT") or "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_MILVUS_TIMEOUT
        except ValueError as exc:
            raise ConfigError(f"MILVUS_TIMEOUT must be a number of seconds, got {raw_timeout!r}") from exc
        return cls(
            uri=(env.get("MILVUS_URI") or DEFAULT_MILVUS_URI).strip(),
            token=(env.get("MILVUS_TOKEN") or "").strip(),
            collection=(collection or env.get("MILVUS_COLLECTION") or DEFAULT_MILVUS_COLLECTION).strip(),
            timeout=timeout,
        )

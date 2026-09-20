"""Conservative CSV validation and cleaning for the search corpus."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REQUIRED_COLUMNS

_WHITESPACE = re.compile(r"\s+")
UNKNOWN_ARTIST = "Unknown artist"
UNTITLED_SONG = "Untitled song"


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be used as the specified corpus."""


def safe_source_url(link: str | None) -> str | None:
    """Only absolute HTTP(S) links become clickable; relative source paths are never turned into URLs."""
    return link if link and link.lower().startswith(("https://", "http://")) else None


@dataclass(frozen=True)
class SongRecord:
    song_id: str
    artist: str
    song: str
    link: str | None
    lyrics: str

    def metadata(self, excerpt_chars: int = 300) -> dict[str, Any]:
        # Keep the original source value; only HTTP(S) links are exposed as clickable URLs.
        safe_url = safe_source_url(self.link)
        excerpt = self.lyrics[:excerpt_chars].strip()
        if len(self.lyrics) > excerpt_chars:
            excerpt += "…"
        return {
            "song_id": self.song_id,
            "artist": self.artist,
            "song": self.song,
            "source_link": self.link,
            "safe_source_url": safe_url,
            "excerpt": excerpt or None,
        }


@dataclass
class PreprocessReport:
    original_rows: int = 0
    missing_or_empty_lyrics: int = 0
    duplicate_rows_removed: int = 0
    valid_cleaned_count: int = 0
    selected_for_build: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def clean_text(value: object) -> str:
    """Normalize whitespace only; do not remove wording, punctuation, or negations."""
    if _is_missing(value):
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def dataset_fingerprint(path: Path) -> str:
    """SHA-256 of the input file, streaming to avoid a second large in-memory copy."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def load_and_clean_csv(path: str | Path) -> tuple[list[SongRecord], PreprocessReport]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DatasetValidationError(
            f"Dataset not found: {csv_path}. Put spotify_millsongdata.csv in data/ or pass --csv PATH."
        )
    try:
        frame = pd.read_csv(csv_path, dtype=object)
    except Exception as exc:
        raise DatasetValidationError(f"Could not read CSV {csv_path}: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetValidationError(f"CSV is missing required columns: {', '.join(missing)}")

    report = PreprocessReport(original_rows=len(frame))
    records: list[SongRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for row in frame.loc[:, REQUIRED_COLUMNS].itertuples(index=False, name=None):
        raw_artist, raw_song, raw_link, raw_lyrics = row
        lyrics = clean_text(raw_lyrics)
        if not lyrics:
            report.missing_or_empty_lyrics += 1
            continue
        artist = clean_text(raw_artist) or UNKNOWN_ARTIST
        song = clean_text(raw_song) or UNTITLED_SONG
        link = clean_text(raw_link) or None
        duplicate_key = (artist, song, lyrics)
        if duplicate_key in seen:
            report.duplicate_rows_removed += 1
            continue
        seen.add(duplicate_key)
        records.append(SongRecord(str(len(records)), artist, song, link, lyrics))
    report.valid_cleaned_count = len(records)
    return records, report


def sample_records(records: list[SongRecord], sample_size: int | None, seed: int) -> list[SongRecord]:
    if sample_size is None:
        return records
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(records):
        return records
    indices = pd.Series(range(len(records))).sample(n=sample_size, random_state=seed).sort_values()
    return [records[int(index)] for index in indices]

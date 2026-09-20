import pandas as pd
import pytest

from src.preprocess import DatasetValidationError, load_and_clean_csv


def write_csv(tmp_path, rows, columns=("artist", "song", "link", "text")):
    path = tmp_path / "songs.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_missing_required_columns_are_rejected(tmp_path):
    path = write_csv(tmp_path, [["a", "s", "lyrics"]], ("artist", "song", "text"))
    with pytest.raises(DatasetValidationError, match="link"):
        load_and_clean_csv(path)


def test_empty_lyrics_duplicates_and_same_title_are_handled(tmp_path):
    path = write_csv(tmp_path, [
        ["A", "Same", "/a", " hello   world "], ["A", "Same", "/a", "hello world"],
        ["B", "Same", "/b", "hello world"], [None, None, None, None], [None, None, None, "still here"],
    ])
    records, report = load_and_clean_csv(path)
    assert report.original_rows == 5
    assert report.missing_or_empty_lyrics == 1
    assert report.duplicate_rows_removed == 1
    assert [(r.artist, r.song, r.lyrics) for r in records] == [
        ("A", "Same", "hello world"), ("B", "Same", "hello world"), ("Unknown artist", "Untitled song", "still here")]

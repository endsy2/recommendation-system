"""Optional lexical comparison baseline; never replaces dense retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import SongRecord


@dataclass
class TfidfBaseline:
    song_ids: list[str]
    matrix: object
    vectorizer: object

    @classmethod
    def from_records(cls, records: list[SongRecord]) -> "TfidfBaseline":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the optional TF-IDF baseline") from exc
        vectorizer = TfidfVectorizer(stop_words=None, dtype=np.float32, norm="l2")
        return cls([record.song_id for record in records], vectorizer.fit_transform([record.lyrics for record in records]), vectorizer)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        values = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-values, kind="stable")[:top_k]
        return [(self.song_ids[int(position)], float(values[int(position)])) for position in order]

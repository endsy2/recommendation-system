import numpy as np
import pytest

from src.config import MilvusSettings
from src.milvus_store import IncompatibleCollectionError, SongRow, import_rows
from src.search_engine import QueryValidationError, SearchEngine, check_embedder_compatibility
from tests.fake_milvus import FakeMilvusClient
from tests.helpers import batches_of, make_manifest


class FakeEmbedder:
    class Info:
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384
        input_limit = 20
        revision = "rev-1"
    info = Info()

    def tokenize(self, text): return text.split()

    def encode_query(self, text):
        vector = np.zeros(384, np.float32)
        vector[0 if "first" in text else 1] = 1
        return vector


def axis(i: int, j: int | None = None) -> np.ndarray:
    vector = np.zeros(384, np.float32)
    vector[i] = 1
    if j is not None:
        vector[j] = 1
    return vector / np.linalg.norm(vector)


@pytest.fixture
def engine():
    client = FakeMilvusClient()
    rows = [SongRow("0", axis(0), "A", "First", "https://example.org/first", "first excerpt"),
            SongRow("1", axis(1), "B", "Second", "/b/second.html", None),
            SongRow("2", axis(0, 1), "C", "Both", None, "both excerpt")]
    manifest = make_manifest("songs_sample", 3)
    manifest["tokenizer_model_input_limit"] = 20
    import_rows(client, manifest, batches_of(rows, 2), [row.song_id for row in rows], 5.0)
    return SearchEngine.from_milvus(MilvusSettings(collection="songs_sample"), embedder=FakeEmbedder(),
                                    client_factory=lambda **_: client)


def test_order_scores_and_fewer_results_than_requested(engine):
    results = engine.search("first", 50)
    assert [r.song_id for r in results] == ["0", "2", "1"]  # only 3 songs exist
    assert [r.similarity_score for r in results] == pytest.approx([1.0, 2 ** -0.5, 0.0], abs=1e-6)
    assert [r.rank for r in results] == [1, 2, 3]
    assert engine.search("second", 1)[0].song_id == "1"


def test_response_mapping_keeps_the_existing_api_shape(engine):
    first, both, second = (r.to_dict() for r in engine.search("first", 3))
    assert list(first) == ["rank", "song_id", "artist", "song", "similarity_score", "source_link", "safe_source_url", "excerpt"]
    assert first == {"rank": 1, "song_id": "0", "artist": "A", "song": "First", "similarity_score": 1.0,
                     "source_link": "https://example.org/first", "safe_source_url": "https://example.org/first",
                     "excerpt": "first excerpt"}
    assert (second["source_link"], second["safe_source_url"], second["excerpt"]) == ("/b/second.html", None, None)
    assert both["source_link"] is None and both["safe_source_url"] is None
    assert isinstance(first["song_id"], str)
    assert 0 < both["similarity_score"] < 1  # raw cosine, not a percentage


def test_bad_queries(engine):
    with pytest.raises(QueryValidationError): engine.search("   ")
    with pytest.raises(QueryValidationError): engine.search("first", 0)
    with pytest.raises(QueryValidationError): engine.search("first", 51)
    with pytest.raises(QueryValidationError): engine.search("first", True)
    with pytest.raises(QueryValidationError, match="too long"): engine.search("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen")


def test_embedder_must_match_the_collection_build():
    manifest = make_manifest("songs_sample", 3)
    check_embedder_compatibility(dict(manifest, tokenizer_model_input_limit=20), FakeEmbedder())
    with pytest.raises(IncompatibleCollectionError, match="revision"):
        check_embedder_compatibility(dict(manifest, tokenizer_model_input_limit=20,
                                          model={**manifest["model"], "resolved_revision": "rev-2"}), FakeEmbedder())
    with pytest.raises(IncompatibleCollectionError, match="input limit"):
        check_embedder_compatibility(manifest, FakeEmbedder())  # collection built with 256 > 20

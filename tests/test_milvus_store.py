import copy

import numpy as np
import pytest

from src.config import MANIFEST_COLLECTION, MilvusSettings
from src.milvus_store import (
    BUILD_COMPLETE,
    BUILD_IMPORTING,
    CollectionNotReadyError,
    IncompatibleCollectionError,
    MilvusSongStore,
    MilvusUnavailableError,
    RecordValidationError,
    SongRow,
    check_collection_matches_build_mode,
    duplicate_ids,
    import_rows,
    index_problems,
    parse_hits,
    read_manifest,
    row_problems,
    schema_problems,
    write_manifest,
)
from tests.fake_milvus import FakeMilvusClient
from tests.helpers import batches_of, make_manifest, make_rows, unit_vectors

TIMEOUT = 5.0


def imported(client, collection="songs_sample", count=5, build_mode="sample", **manifest_kwargs):
    rows = make_rows(count)
    manifest = make_manifest(collection, count, build_mode, **manifest_kwargs)
    result = import_rows(client, manifest, batches_of(rows, 2), [row.song_id for row in rows], TIMEOUT)
    return rows, result


# ----------------------------------------------------------------------------- schema and index

def test_created_collection_has_exact_schema_and_hnsw_cosine_index():
    client = FakeMilvusClient()
    imported(client)
    description = client.describe_collection("songs_sample")
    assert schema_problems(description) == []
    fields = {field["name"]: field for field in description["fields"]}
    assert fields["song_id"]["is_primary"] and not fields["song_id"].get("auto_id")
    assert fields["vector"]["params"]["dim"] == 384
    assert fields["link"].get("nullable") and fields["lyrics_excerpt"].get("nullable")
    (index,) = [client.describe_index("songs_sample", name) for name in client.list_indexes("songs_sample")]
    assert (index["field_name"], index["index_type"], index["metric_type"]) == ("vector", "HNSW", "COSINE")
    assert index_problems([index]) == []


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["fields"][1]["params"].update(dim=768), "dim=768"),
    (lambda d: d.update(enable_dynamic_field=True), "dynamic"),
    (lambda d: d["fields"][0].update(auto_id=True), "auto_id"),
    (lambda d: d["fields"].pop(), "missing fields: lyrics_excerpt"),
    (lambda d: d["fields"].append({"name": "genre", "type": "VARCHAR", "params": {"max_length": 9}}), "unexpected fields: genre"),
    (lambda d: d["fields"][0].update(type="INT64"), "type INT64"),
    (lambda d: d["fields"][4].pop("nullable"), "nullable=False"),
    (lambda d: d["fields"][2]["params"].update(max_length=16), "max_length=16"),
])
def test_schema_validation_reports_incompatibilities(mutate, message):
    client = FakeMilvusClient()
    imported(client)
    description = copy.deepcopy(client.describe_collection("songs_sample"))
    mutate(description)
    assert any(message in problem for problem in schema_problems(description))


@pytest.mark.parametrize("index", [
    {"field_name": "vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
    {"field_name": "vector", "index_type": "FLAT", "metric_type": "COSINE"},
    {"field_name": "vector", "index_type": "IVF_FLAT", "metric_type": "COSINE"},
    {"field_name": "vector", "index_type": "HNSW", "metric_type": "L2"},
    {"field_name": "vector", "index_type": "HNSW", "metric_type": "IP"},
])
def test_only_hnsw_cosine_is_accepted(index):
    assert index_problems([index])
    assert index_problems([]) == ["no index on field vector; expected HNSW with COSINE"]


def test_incompatible_existing_collection_is_rejected_and_never_dropped():
    client = FakeMilvusClient()
    imported(client)
    client.collections["songs_sample"]["indexes"]["vector_hnsw_cosine"]["metric_type"] = "L2"
    rows = make_rows(5)
    before = len(client.calls)
    with pytest.raises(IncompatibleCollectionError, match="metric is L2"):
        import_rows(client, make_manifest("songs_sample", 5), batches_of(rows, 2), [r.song_id for r in rows], TIMEOUT)
    assert "songs_sample" in client.collections
    assert not any(call[0] in ("drop_collection", "create_collection", "upsert") for call in client.calls[before:])


def test_existing_collection_without_manifest_is_refused():
    client = FakeMilvusClient()
    imported(client, collection="songs_other")
    client.collections["songs_unknown"] = copy.deepcopy(client.collections["songs_other"])
    rows = make_rows(5)
    with pytest.raises(IncompatibleCollectionError, match="no song-search build manifest"):
        import_rows(client, make_manifest("songs_unknown", 5), batches_of(rows, 2), [r.song_id for r in rows], TIMEOUT)


# ----------------------------------------------------------------------------- record validation

def test_row_validation_rejects_bad_vectors_and_never_truncates_text():
    good = make_rows(1)[0]
    assert row_problems(good) == []
    assert "shape" in row_problems(SongRow("1", np.ones(383, np.float32) / np.sqrt(383), "a", "s", None, None))[0]
    nan = good.vector.copy(); nan[0] = np.nan
    assert "non-finite" in row_problems(SongRow("1", nan, "a", "s", None, None))[0]
    assert "not L2-normalized" in row_problems(SongRow("1", good.vector * 2, "a", "s", None, None))[0]
    long_artist = "é" * 300  # 600 UTF-8 bytes > 512
    row = SongRow("1", good.vector, long_artist, "s", None, None)
    assert "600 UTF-8 bytes" in row_problems(row)[0] and row.artist == long_artist
    assert "artist is missing" in row_problems(SongRow("1", good.vector, None, "s", None, None))[0]
    assert "song is empty" in row_problems(SongRow("1", good.vector, "a", "  ", None, None))[0]
    assert row_problems(SongRow("1", good.vector, "a", "s", None, None)) == []  # null link/excerpt allowed


def test_duplicate_ids_are_detected_before_any_write():
    assert duplicate_ids(["1", "2", "1", "3", "2"]) == ["1", "2"]
    client = FakeMilvusClient()
    rows = make_rows(3)
    rows.append(rows[0])
    with pytest.raises(RecordValidationError, match="Duplicate"):
        import_rows(client, make_manifest("songs_sample", 4), batches_of(rows, 2), [r.song_id for r in rows], TIMEOUT)
    assert client.collections == {}


# ----------------------------------------------------------------------------- import behaviour

def test_import_maps_every_vector_to_its_song_and_marks_build_complete():
    client = FakeMilvusClient()
    rows, result = imported(client, count=7)
    assert result.imported_count == 7 and result.created_collection
    stored = client.collections["songs_sample"]["rows"]
    for row in rows:
        assert np.allclose(stored[row.song_id]["vector"], row.vector, atol=0)
        assert (stored[row.song_id]["artist"], stored[row.song_id]["link"]) == (row.artist, row.link)
    manifest = read_manifest(client, "songs_sample", TIMEOUT)
    assert manifest["build_state"] == BUILD_COMPLETE and manifest["imported_song_count"] == 7
    assert manifest["verification"]["id_set_matches"] and manifest["verification"]["records_read_back"] == 7
    # Verification reads must use Strong consistency so completed writes are visible.
    assert all(call[2].get("consistency_level") == "Strong" for call in client.calls
               if call[0] in ("query", "get", "query_iterator") and call[1] == "songs_sample")


def test_repeat_import_is_idempotent_without_duplicate_logical_records():
    client = FakeMilvusClient()
    rows, _ = imported(client, count=5)
    _, again = imported(client, count=5)
    assert not again.created_collection
    assert len(client.collections["songs_sample"]["rows"]) == 5
    assert again.manifest["build_state"] == BUILD_COMPLETE
    upserts = [call for call in client.calls if call[0] == "upsert" and call[1] == "songs_sample"]
    assert sum(call[2]["rows"] for call in upserts) == 10  # 5 rows written twice, stored once


def test_stale_records_fail_verification_and_leave_build_importing():
    client = FakeMilvusClient()
    imported(client, count=5)
    stale = make_rows(1, seed=9, first_id=999)[0]
    client.collections["songs_sample"]["rows"][stale.song_id] = stale.entity()
    rows = make_rows(5)
    with pytest.raises(RecordValidationError, match="unexpected=\\['999'\\]"):
        import_rows(client, make_manifest("songs_sample", 5), batches_of(rows, 2), [r.song_id for r in rows], TIMEOUT)
    assert read_manifest(client, "songs_sample", TIMEOUT)["build_state"] == BUILD_IMPORTING


def test_different_dataset_or_model_requires_a_new_collection():
    client = FakeMilvusClient()
    imported(client, count=5)
    rows = make_rows(5)
    for kwargs, field in (({"fingerprint": "sha256:other"}, "dataset_fingerprint"), ({"revision": "rev-2"}, "model_revision")):
        with pytest.raises(IncompatibleCollectionError, match=field):
            import_rows(client, make_manifest("songs_sample", 5, **kwargs), batches_of(rows, 2),
                        [r.song_id for r in rows], TIMEOUT)
    assert read_manifest(client, "songs_sample", TIMEOUT)["build_state"] == BUILD_COMPLETE


def test_sample_and_full_builds_stay_in_separate_collections():
    check_collection_matches_build_mode("songs_sample", "sample")
    check_collection_matches_build_mode("songs_full", "full")
    with pytest.raises(RecordValidationError):
        check_collection_matches_build_mode("songs_full", "sample")
    with pytest.raises(RecordValidationError):
        check_collection_matches_build_mode("songs_sample", "full")
    client = FakeMilvusClient()
    imported(client, collection="songs_sample", count=4, build_mode="sample")
    imported(client, collection="songs_full", count=6, build_mode="full")
    assert len(client.collections["songs_sample"]["rows"]) == 4
    assert len(client.collections["songs_full"]["rows"]) == 6
    # Under a neutral name the name guard cannot help; the manifest signature still refuses the mix.
    imported(client, collection="songs_dev", count=4, build_mode="sample")
    rows = make_rows(6)
    with pytest.raises(IncompatibleCollectionError, match="build_mode"):
        import_rows(client, make_manifest("songs_dev", 6, "full"), batches_of(rows, 3), [r.song_id for r in rows], TIMEOUT)
    assert len(client.collections["songs_dev"]["rows"]) == 4


# ----------------------------------------------------------------------------- runtime store

def open_store(client, collection="songs_sample"):
    def reconnect(**_):
        client.closed = False  # a failed open closes its client; model a fresh connection to the same data
        return client

    return MilvusSongStore.open(MilvusSettings(collection=collection, timeout=TIMEOUT), client_factory=reconnect)


def test_store_opens_only_completed_loaded_collections():
    client = FakeMilvusClient()
    with pytest.raises(CollectionNotReadyError, match="does not exist"):
        open_store(client)
    assert client.closed  # a failed open releases its connection
    client.closed = False
    rows, _ = imported(client, count=5)
    manifest = read_manifest(client, "songs_sample", TIMEOUT)
    write_manifest(client, dict(manifest, build_state=BUILD_IMPORTING), TIMEOUT)
    with pytest.raises(CollectionNotReadyError, match="importing"):
        open_store(client)
    client.closed = False
    write_manifest(client, manifest, TIMEOUT)
    client.collections["songs_sample"]["loaded"] = False
    store = open_store(client)  # loads it: non-destructive
    assert client.collections["songs_sample"]["loaded"] and store.song_count == 5
    del client.collections["songs_sample"]["rows"][rows[0].song_id]
    with pytest.raises(CollectionNotReadyError, match="holds 4 songs"):
        store.check_ready()


def test_store_rejects_manifest_registry_with_wrong_schema():
    client = FakeMilvusClient()
    imported(client)
    client.collections[MANIFEST_COLLECTION]["description"]["fields"].pop()
    with pytest.raises(IncompatibleCollectionError, match="not a song-search manifest registry"):
        open_store(client)


def test_unavailable_server_gives_setup_hint_without_credentials():
    def refuse(**kwargs):
        raise ConnectionError(f"cannot reach {kwargs['uri']} with {kwargs['token']}")

    settings = MilvusSettings(uri="http://admin:pw@localhost:19530", token="secret-token")
    with pytest.raises(MilvusUnavailableError) as info:
        MilvusSongStore.open(settings, client_factory=refuse)
    message = str(info.value)
    assert "docker compose" in message and "localhost:19530" in message
    assert "secret-token" not in message and "pw" not in message


def test_outage_after_startup_becomes_not_ready_error():
    client = FakeMilvusClient()
    imported(client)
    store = open_store(client)
    client.fail = TimeoutError("deadline exceeded")
    with pytest.raises(MilvusUnavailableError, match="search songs"):
        store.search(unit_vectors(1)[0], 3)


def test_search_uses_vector_field_cosine_limit_and_output_fields():
    client = FakeMilvusClient()
    rows, _ = imported(client, count=5)
    store = open_store(client)
    hits = store.search(rows[2].vector, 3)
    call = [c for c in client.calls if c[0] == "search"][-1][2]
    assert call["anns_field"] == "vector" and call["limit"] == 3
    assert call["search_params"]["metric_type"] == "COSINE"
    assert call["search_params"]["params"] == {"ef": 64}
    assert call["output_fields"] == ["artist", "song", "link", "lyrics_excerpt"]
    assert hits[0].song_id == rows[2].song_id and hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert (hits[0].artist, hits[0].song, hits[0].link, hits[0].lyrics_excerpt) == (
        rows[2].artist, rows[2].song, rows[2].link, rows[2].lyrics_excerpt)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert len(store.search(rows[0].vector, 50)) == 5  # fewer songs than requested


def test_parse_hits_accepts_pk_or_id_and_distance_or_score():
    hits = parse_hits([
        {"song_id": "7", "distance": 0.2, "entity": {"artist": "A", "song": "S", "link": None, "lyrics_excerpt": "x"}},
        {"id": "8", "score": 0.9, "entity": {"artist": "B", "song": "T", "link": "/b", "lyrics_excerpt": None}},
    ])
    assert [(hit.song_id, hit.score) for hit in hits] == [("8", 0.9), ("7", 0.2)]
    with pytest.raises(CollectionNotReadyError):
        parse_hits([{"entity": {}}])

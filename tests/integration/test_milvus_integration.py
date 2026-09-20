"""Integration tests against a real Milvus server (opt-in).

Run with MILVUS_INTEGRATION=1 and a server reachable at MILVUS_URI (default http://localhost:19530).
Each test uses uniquely named throwaway collections and drops only those; nothing else is touched.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import replace

import numpy as np
import pytest

from src.config import MANIFEST_COLLECTION, MilvusSettings
from src.milvus_store import (
    BUILD_COMPLETE,
    IncompatibleCollectionError,
    MilvusSongStore,
    count_entities,
    describe_indexes,
    import_rows,
    index_problems,
    read_manifest,
    schema_problems,
)
from src.verify_migration import compare_rankings
from tests.helpers import batches_of, make_manifest, make_rows, unit_vectors

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("MILVUS_INTEGRATION") != "1", reason="set MILVUS_INTEGRATION=1 to run against Milvus"),
]


@pytest.fixture
def settings():
    return MilvusSettings.from_env()


@pytest.fixture
def client(settings):
    from src.milvus_store import connect

    client = connect(settings)
    created: list[str] = []
    client.created = created
    yield client
    for name in created:
        if client.has_collection(name):
            client.drop_collection(name)
    if created and client.has_collection(MANIFEST_COLLECTION):
        client.delete(MANIFEST_COLLECTION, ids=created)
    client.close()


def fresh_name(client, kind="sample") -> str:
    name = f"it_songs_{kind}_{uuid.uuid4().hex[:10]}"
    client.created.append(name)
    return name


def rows_with_ties(count: int):
    rows = make_rows(count, seed=3)
    # Two exact duplicates of song 0's vector create a genuine tie for queries near it.
    rows[5] = replace(rows[5], vector=rows[0].vector.copy())
    rows[6] = replace(rows[6], vector=rows[0].vector.copy())
    return rows


def test_real_server_schema_index_import_and_idempotent_upsert(client, settings):
    name = fresh_name(client)
    rows = rows_with_ties(40)
    manifest = make_manifest(name, len(rows))
    result = import_rows(client, manifest, batches_of(rows, 16), [r.song_id for r in rows], settings.timeout)
    assert result.created_collection and result.imported_count == 40
    # The server's own description must match the explicit schema and HNSW/COSINE index.
    assert schema_problems(client.describe_collection(name)) == []
    indexes = describe_indexes(client, name, settings.timeout)
    assert index_problems(indexes) == [] and indexes[0]["index_type"] == "HNSW" and indexes[0]["metric_type"] == "COSINE"
    # Re-running the same import keeps one logical record per song ID.
    again = import_rows(client, manifest, batches_of(rows, 7), [r.song_id for r in rows], settings.timeout)
    assert not again.created_collection
    assert count_entities(client, name, settings.timeout) == 40
    stored = read_manifest(client, name, settings.timeout)
    assert stored["build_state"] == BUILD_COMPLETE and stored["verification"]["records_read_back"] == 40


def test_milvus_hnsw_cosine_matches_faiss_flat_ip_on_same_vectors(client, settings):
    faiss = pytest.importorskip("faiss")
    name = fresh_name(client)
    rows = rows_with_ties(60)
    import_rows(client, make_manifest(name, len(rows)), batches_of(rows, 25), [r.song_id for r in rows], settings.timeout)
    store = MilvusSongStore.open(settings.with_collection(name))
    try:
        matrix = np.vstack([row.vector for row in rows]).astype(np.float32)
        ids = [row.song_id for row in rows]
        index = faiss.IndexFlatIP(384)
        index.add(matrix)
        queries = [rows[0].vector, rows[10].vector, *unit_vectors(8, seed=11)]
        for number, query in enumerate(queries):
            scores, positions = index.search(query.reshape(1, -1), 10)
            hits = store.search(query, 10, consistency_level="Strong")
            exact = {hit.song_id: float(matrix[ids.index(hit.song_id)] @ query) for hit in hits}
            comparison = compare_rankings(f"q{number}", [ids[p] for p in positions[0]], scores[0].tolist(),
                                          [hit.song_id for hit in hits], [hit.score for hit in hits], exact)
            assert comparison.passed, comparison.problems
        # Query equal to song 0 ties exactly with songs 5 and 6: all three must lead, in any order.
        top3 = {hit.song_id for hit in store.search(rows[0].vector, 3, consistency_level="Strong")}
        assert top3 == {rows[0].song_id, rows[5].song_id, rows[6].song_id}
    finally:
        store.close()


def test_metadata_retrieval_nulls_and_fewer_results_than_limit(client, settings):
    name = fresh_name(client)
    rows = make_rows(4, seed=5)  # rows[0] and rows[3] have link=None
    import_rows(client, make_manifest(name, 4), batches_of(rows, 4), [r.song_id for r in rows], settings.timeout)
    store = MilvusSongStore.open(settings.with_collection(name))
    try:
        hits = store.search(rows[3].vector, 50)
        assert len(hits) == 4
        top = hits[0]
        assert (top.song_id, top.artist, top.song, top.link, top.lyrics_excerpt) == (
            rows[3].song_id, rows[3].artist, rows[3].song, None, rows[3].lyrics_excerpt)
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
        assert {row["song_id"] for row in store.iter_metadata()} == {row.song_id for row in rows}
    finally:
        store.close()


def test_incompatible_existing_collection_is_rejected_and_kept(client, settings):
    from pymilvus import DataType, MilvusClient

    name = fresh_name(client)
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("song_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2", params={})
    client.create_collection(name, schema=schema, index_params=index_params)
    rows = make_rows(3)
    with pytest.raises(IncompatibleCollectionError, match="dim=8"):
        import_rows(client, make_manifest(name, 3), batches_of(rows, 3), [r.song_id for r in rows], settings.timeout)
    assert client.has_collection(name)
    with pytest.raises(IncompatibleCollectionError):
        MilvusSongStore.open(settings.with_collection(name))


def test_unreachable_server_fails_fast_with_setup_hint():
    from src.milvus_store import MilvusUnavailableError

    bad = MilvusSettings(uri="http://127.0.0.1:1", timeout=2)
    with pytest.raises(MilvusUnavailableError, match="docker compose"):
        MilvusSongStore.open(bad)

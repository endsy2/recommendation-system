import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src import migrate_faiss_to_milvus as migrate
from src.faiss_artifacts import ArtifactExportError, load_faiss_artifact, validate_artifact_rows
from src.milvus_store import BUILD_COMPLETE, read_manifest
from src.verify_migration import compare_rankings
from tests.fake_milvus import FakeMilvusClient
from tests.helpers import make_rows, write_faiss_artifact


def digest(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())}


@pytest.fixture
def artifact_dir(tmp_path):
    return write_faiss_artifact(tmp_path / "sample", make_rows(12))


@pytest.fixture
def fake(monkeypatch):
    client = FakeMilvusClient()

    def reconnect(settings):
        client.closed = False
        return client

    monkeypatch.setattr(migrate, "connect", reconnect)
    for key in ("MILVUS_URI", "MILVUS_TOKEN", "MILVUS_COLLECTION", "MILVUS_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    return client


# ----------------------------------------------------------------------------- source artifact

def test_iter_rows_pairs_each_stored_vector_with_its_own_metadata(artifact_dir):
    artifact = load_faiss_artifact(artifact_dir)
    assert artifact.index_type == "IndexFlatIP" and artifact.count == 12
    rows = [row for batch in artifact.iter_rows(5) for row in batch]  # batches of 5, 5, 2
    expected = make_rows(12)
    assert [row.song_id for row in rows] == [row.song_id for row in expected]
    for row, source in zip(rows, expected):
        assert np.array_equal(row.vector, source.vector)
        assert (row.artist, row.song, row.link, row.lyrics_excerpt) == (source.artist, source.song, source.link, source.lyrics_excerpt)
    assert validate_artifact_rows(artifact, 5)["records"] == 12


@pytest.mark.parametrize("factory, name", [
    (lambda: faiss.IndexFlatL2(384), "IndexFlatL2"),
    (lambda: faiss.IndexHNSWFlat(384, 8, faiss.METRIC_INNER_PRODUCT), "IndexHNSWFlat"),
    (lambda: faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT), "IndexScalarQuantizer"),
])
def test_unsupported_index_types_fail_clearly(tmp_path, factory, name):
    directory = write_faiss_artifact(tmp_path / name, make_rows(12), index_factory=factory)
    with pytest.raises(ArtifactExportError, match=f"Unsupported FAISS index type {name}"):
        load_faiss_artifact(directory)


def test_untrusted_or_inconsistent_artifacts_are_refused(artifact_dir):
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(dict(manifest, faiss_index_file_bytes=1)))
    with pytest.raises(ArtifactExportError, match="does not match its build record"):
        load_faiss_artifact(artifact_dir)
    manifest_path.write_text(json.dumps(dict(manifest, model={"identifier": "other/model"})))
    with pytest.raises(ArtifactExportError, match="not a trusted project artifact"):
        load_faiss_artifact(artifact_dir)
    manifest_path.write_text(json.dumps(manifest))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata[:-1]))
    with pytest.raises(ArtifactExportError, match="Count mismatch"):
        load_faiss_artifact(artifact_dir)
    (artifact_dir / "metadata.json").unlink()
    with pytest.raises(ArtifactExportError, match="incomplete"):
        load_faiss_artifact(artifact_dir)


def test_invalid_vectors_and_duplicate_ids_are_caught_before_writing(tmp_path):
    rows = make_rows(4)
    bad = write_faiss_artifact(tmp_path / "bad", [rows[0], replace(rows[1], vector=rows[1].vector * 3)])
    with pytest.raises(ArtifactExportError, match="not L2-normalized"):
        validate_artifact_rows(load_faiss_artifact(bad), 10)
    dup = write_faiss_artifact(tmp_path / "dup", [rows[0], rows[0]])
    with pytest.raises(ArtifactExportError, match="Duplicate song IDs"):
        validate_artifact_rows(load_faiss_artifact(dup), 10)


# ----------------------------------------------------------------------------- migration command

def test_dry_run_validates_without_writing(artifact_dir, fake, capsys):
    before = digest(artifact_dir)
    assert migrate.main(["--artifacts", str(artifact_dir), "--collection", "songs_sample", "--dry-run"]) == 0
    assert "Nothing was written" in capsys.readouterr().out
    assert fake.collections == {}
    assert not any(call[0] in ("upsert", "create_collection") for call in fake.calls)
    assert digest(artifact_dir) == before


def test_migration_reuses_vectors_is_repeatable_and_keeps_backups(artifact_dir, fake, capsys):
    before = digest(artifact_dir)
    args = ["--artifacts", str(artifact_dir), "--collection", "songs_sample", "--batch-size", "5"]
    assert migrate.main(args) == 0
    assert migrate.main(args) == 0  # safe to re-run
    output = capsys.readouterr().out
    assert "Migrated 12 songs" in output and "embeddings reused" in output
    stored = fake.collections["songs_sample"]["rows"]
    assert len(stored) == 12
    for row in make_rows(12):
        assert np.array_equal(np.asarray(stored[row.song_id]["vector"], np.float32), row.vector)
    assert fake.closed  # the command releases its connection
    fake.closed = False
    manifest = read_manifest(fake, "songs_sample", 5.0)
    assert manifest["build_state"] == BUILD_COMPLETE and manifest["imported_song_count"] == 12
    assert manifest["source"]["faiss_index_type"] == "IndexFlatIP"
    assert manifest["embeddings"].startswith("reused")
    assert digest(artifact_dir) == before  # source files untouched


def test_sample_artifact_cannot_be_migrated_into_full_collection(artifact_dir, fake, capsys):
    assert migrate.main(["--artifacts", str(artifact_dir), "--collection", "songs_full"]) == migrate.EXIT_INVALID
    assert "Refusing to write a sample build" in capsys.readouterr().err
    assert fake.collections == {}


def test_unreachable_milvus_is_reported_not_bypassed(artifact_dir, monkeypatch, capsys):
    from src.milvus_store import MilvusUnavailableError

    def refuse(settings):
        raise MilvusUnavailableError("Could not connect to Milvus at http://localhost:19530")

    monkeypatch.setattr(migrate, "connect", refuse)
    assert migrate.main(["--artifacts", str(artifact_dir), "--collection", "songs_sample"]) == migrate.EXIT_UNAVAILABLE
    assert "Could not connect" in capsys.readouterr().err


# ----------------------------------------------------------------------------- baseline comparison

def test_compare_rankings_tolerates_ties_but_not_score_or_membership_errors():
    ids, scores = ["a", "b", "c", "d"], [0.9, 0.7, 0.7, 0.5]
    exact = {"a": 0.9, "b": 0.7, "c": 0.7, "d": 0.5, "e": 0.5}
    same = compare_rankings("q", ids, scores, ids, scores, exact)
    assert same.passed and same.exact_id_order_matches
    swapped = compare_rankings("q", ids, scores, ["a", "c", "b", "d"], scores, exact)
    assert swapped.passed and not swapped.exact_id_order_matches  # b/c are exactly tied
    boundary = compare_rankings("q", ids, scores, ["a", "b", "c", "e"], scores, exact)
    assert boundary.passed  # d and e tie at the k-th score
    drift = compare_rankings("q", ids, scores, ids, [0.9, 0.7, 0.7, 0.49], exact)
    assert not drift.passed
    missing = compare_rankings("q", ids, scores, ["a", "e", "c", "d"], [0.9, 0.5, 0.7, 0.5], exact)
    assert not missing.passed and any("missing non-tied" in p for p in missing.problems)
    top = compare_rankings("q", ids, scores, ["b", "a", "c", "d"], [0.7, 0.9, 0.7, 0.5], exact)
    assert not top.top1_agrees

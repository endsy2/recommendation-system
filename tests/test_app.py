import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.milvus_store import CollectionNotReadyError, MilvusUnavailableError
from src.search_engine import SearchResult


class FakeStore:
    collection = "songs_sample"

    def __init__(self):
        self.error = None

    def check_ready(self, allow_load=False):
        if self.error:
            raise self.error
        return {"imported_song_count": 2, "build_mode": "sample"}


class FakeEngine:
    def __init__(self):
        self.store, self.closed, self.error = FakeStore(), False, None
        self.collection = "songs_sample"

    def search_with_metrics(self, query, top_k):
        if self.error:
            raise self.error
        results = [SearchResult(1, "7", "A", "Song", 0.81234567, "/a/song.html", None, "excerpt")][:top_k]
        return results, {}

    def close(self):
        self.closed = True


@pytest.fixture
def env(monkeypatch):
    for key, value in {"MILVUS_URI": "http://localhost:19530", "MILVUS_TOKEN": "tok-123",
                       "MILVUS_COLLECTION": "songs_sample", "MILVUS_TIMEOUT": "3"}.items():
        monkeypatch.setenv(key, value)


def client_with(factory):
    app_module.app.state.engine_factory = factory
    return TestClient(app_module.app)


def test_ready_engine_serves_existing_contract_and_is_closed_on_shutdown(env):
    engine = FakeEngine()
    with client_with(lambda settings: engine) as client:
        assert client.get("/health").json() == {"ready": True, "indexed_song_count": 2, "artifact_mode": "sample",
                                                 "collection": "songs_sample"}
        body = client.post("/search", json={"query": "  lonely  ", "top_k": 5}).json()
        assert body == {"query": "lonely", "results": [{"rank": 1, "song_id": "7", "artist": "A", "song": "Song",
                        "similarity_score": 0.812346, "source_link": "/a/song.html", "safe_source_url": None,
                        "excerpt": "excerpt"}]}
        assert client.post("/search", json={"query": "   "}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 51}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
        assert "Results are based on lyrics and inferred themes, not audio analysis." in client.get("/").text
    assert engine.closed


def test_database_unavailable_at_startup_returns_503_without_credentials(env):
    def refuse(settings):
        raise MilvusUnavailableError(f"Could not connect to Milvus at {settings.safe_uri}. Start it with docker compose.")

    with client_with(refuse) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and "Could not connect" in health["detail"]
        response = client.post("/search", json={"query": "hello"})
        assert response.status_code == 503 and "docker compose" in response.json()["detail"]
        assert "tok-123" not in response.text and "tok-123" not in str(health)


def test_collection_not_ready_after_startup_is_reported(env):
    engine = FakeEngine()
    with client_with(lambda settings: engine) as client:
        engine.store.error = CollectionNotReadyError("Collection 'songs_sample' build is 'importing', not 'complete'.")
        assert client.get("/health").json() == {"ready": False, "detail": engine.store.error.args[0]}
        engine.error = MilvusUnavailableError("Milvus request failed while trying to search songs (TimeoutError).")
        response = client.post("/search", json={"query": "hello"})
        assert response.status_code == 503 and "search songs" in response.json()["detail"]


def test_retries_initialization_only_while_not_ready(env, monkeypatch):
    attempts, engine = [], FakeEngine()

    def flaky(settings):
        attempts.append(1)
        if len(attempts) == 1:
            raise MilvusUnavailableError("down")
        return engine

    monkeypatch.setattr(app_module, "RETRY_SECONDS", 0.0)
    with client_with(flaky) as client:
        assert client.post("/search", json={"query": "hello"}).status_code == 200
        client.post("/search", json={"query": "hello"})
        client.get("/health")
    assert len(attempts) == 2  # never reconnects once ready


def test_invalid_configuration_is_reported(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "./songs.db")
    with client_with(lambda settings: FakeEngine()) as client:
        detail = client.get("/health").json()["detail"]
        assert "Invalid Milvus configuration" in detail and "Milvus Lite" in detail
        assert client.post("/search", json={"query": "x"}).status_code == 503

import pytest

from src.config import (
    DEFAULT_MILVUS_COLLECTION,
    DEFAULT_MILVUS_TIMEOUT,
    DEFAULT_MILVUS_URI,
    MANIFEST_COLLECTION,
    ConfigError,
    MilvusSettings,
    redact_uri,
)


def test_defaults_from_empty_environment():
    settings = MilvusSettings.from_env({})
    assert settings.uri == DEFAULT_MILVUS_URI == "http://localhost:19530"
    assert settings.collection == DEFAULT_MILVUS_COLLECTION == "songs_sample"
    assert settings.timeout == DEFAULT_MILVUS_TIMEOUT
    assert settings.token == ""


def test_environment_values_and_explicit_collection_override():
    env = {"MILVUS_URI": "https://milvus.example.org:443", "MILVUS_TOKEN": "user:secret",
           "MILVUS_COLLECTION": "songs_full", "MILVUS_TIMEOUT": "2.5"}
    settings = MilvusSettings.from_env(env)
    assert (settings.uri, settings.collection, settings.timeout) == ("https://milvus.example.org:443", "songs_full", 2.5)
    assert MilvusSettings.from_env(env, collection="songs_sample").collection == "songs_sample"
    assert settings.with_collection("songs_sample").token == "user:secret"


@pytest.mark.parametrize("timeout", ["abc", "0", "-1", "301"])
def test_invalid_timeouts_are_rejected(timeout):
    with pytest.raises(ConfigError, match="MILVUS_TIMEOUT"):
        MilvusSettings.from_env({"MILVUS_TIMEOUT": timeout})


@pytest.mark.parametrize("uri", ["./milvus_demo.db", "file:///tmp/songs.db", "ftp://localhost:19530", "http://", "http://localhost:notaport"])
def test_non_server_uris_are_rejected_without_lite_fallback(uri):
    with pytest.raises(ConfigError):
        MilvusSettings.from_env({"MILVUS_URI": uri})


def test_lite_database_file_gets_a_specific_hint():
    with pytest.raises(ConfigError, match="Milvus Lite"):
        MilvusSettings.from_env({"MILVUS_URI": "songs.db"})


@pytest.mark.parametrize("name", ["1songs", "songs-sample", "songs sample", "x" * 256, MANIFEST_COLLECTION])
def test_invalid_collection_names_are_rejected(name):
    with pytest.raises(ConfigError):
        MilvusSettings.from_env({"MILVUS_COLLECTION": name})


def test_credentials_never_appear_in_repr_or_safe_uri():
    settings = MilvusSettings(uri="http://root:pw@localhost:19530", token="root:s3cret")
    assert "s3cret" not in repr(settings) and "pw" not in repr(settings) and "pw" not in settings.safe_uri
    assert redact_uri("http://root:pw@localhost:19530") == "http://localhost:19530"
    assert redact_uri(DEFAULT_MILVUS_URI) == DEFAULT_MILVUS_URI

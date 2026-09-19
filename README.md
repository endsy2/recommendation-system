# Lyrics semantic song search

A runnable, lyrics-only semantic search engine. Describe a feeling or theme such as `lonely and reflective after a breakup`; it returns songs whose **lyrics** have nearby sentence-transformer embeddings. It does not measure tempo, instruments, acoustic energy, or actual musical genre.

> Results are based on lyrics and inferred themes, not audio analysis.

Vectors and song metadata are stored in and searched by **Milvus** (Standalone, HNSW index, COSINE metric) through `pymilvus.MilvusClient`. The earlier FAISS `IndexFlatIP` artifacts are kept as read-only backups and as the baseline that the Milvus results are verified against; FAISS is no longer on the web app's search path.

## Dataset and privacy

The required Kaggle dataset is `spotify_millsongdata.csv` from [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset). It must contain `artist`, `song`, `link`, and `text`. The commands below use `data/spotify_millsongdata.csv`; any path can be given with `--csv` (the built-in default, `dataset/archive/spotify_millsongdata.csv`, only applies when `--csv` is omitted).

Datasets, generated indexes, Milvus data volumes, model caches, `.env`, and lyric text are ignored by Git. The `link` field is retained as source metadata; only an existing absolute `http://` or `https://` value is rendered as a clickable link. Relative source paths are never turned into guessed playback URLs.

## Versions

| Component | Version | Where it is pinned |
| --- | --- | --- |
| Milvus Standalone server | `milvusdb/milvus:v3.0.1` | `deploy/milvus/docker-compose.yml` |
| Python SDK | `pymilvus==3.0.1` | `requirements.txt` |
| etcd / MinIO (Milvus dependencies) | `v3.5.25` / `RELEASE.2024-12-18T13-15-44Z` | `deploy/milvus/docker-compose.yml` |

The Milvus v3.0.1 release notes list Python SDK 3.0.1 for that server. The compose file is the official v3.0.1 `milvus-standalone-docker-compose.yml` with four documented changes: ports bound to `127.0.0.1`, a fixed project name, a 60 s stop grace period (Docker's 10 s default killed Milvus with exit 137 during testing), and the MinIO image pulled from MinIO's `quay.io` registry because the Docker Hub reference in the official file returned "pull access denied". Installing `pymilvus` does **not** start a database; Milvus runs in Docker.

## Setup

The project path contains spaces, so keep paths quoted. CPU is supported; GPU is optional.

PowerShell (Windows):

```powershell
cd "D:\data science and analysis\data analysis with python\deep learning\week2"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env if needed; never commit it
```

Bash (macOS/Linux/Git Bash/WSL):

```bash
cd "/path/to/week2"
python3 -m venv .venv
source .venv/bin/activate          # Git Bash on Windows: source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env               # then edit .env if needed; never commit it
```

### Configuration

Settings come from environment variables; a project `.env` file is also read, but real environment variables always win.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MILVUS_URI` | `http://localhost:19530` | Milvus server address (`http(s)://` or `tcp://`). File paths / Milvus Lite `.db` files are rejected. |
| `MILVUS_TOKEN` | empty | `user:password` or API key for an authenticated server. The local compose server has auth disabled. |
| `MILVUS_COLLECTION` | `songs_sample` | The completed collection the web app serves. |
| `MILVUS_TIMEOUT` | `10` | Per-request timeout in seconds (0 < value ≤ 300). |
| `SONG_SEARCH_DEVICE` | empty | Optional model device, e.g. `cpu`. |

There is no fallback: if the configured server or collection is unavailable, commands exit with a setup hint and the API reports not-ready (HTTP 503) instead of silently using Milvus Lite or FAISS. `SONG_SEARCH_ARTIFACTS` is no longer used.

## Start and stop Milvus

Requires Docker Desktop (Windows: with the WSL 2 backend) or Docker Engine with Compose v2. Data persists in `deploy/milvus/volumes/` (bind mounts; override the parent directory with `DOCKER_VOLUME_DIRECTORY`). Services listen on `127.0.0.1` only: gRPC `19530`, health/metrics `9091`, MinIO `9000`/`9001` (MinIO uses the official local-development credentials `minioadmin`, reachable only from this machine).

PowerShell:

```powershell
docker compose -f deploy/milvus/docker-compose.yml up -d        # start (first run pulls images)
curl.exe http://127.0.0.1:9091/healthz                          # prints OK when ready
docker compose -f deploy/milvus/docker-compose.yml ps
docker compose -f deploy/milvus/docker-compose.yml stop         # stop, keep data
docker compose -f deploy/milvus/docker-compose.yml start        # start again
```

Bash:

```bash
docker compose -f deploy/milvus/docker-compose.yml up -d
curl http://127.0.0.1:9091/healthz
docker compose -f deploy/milvus/docker-compose.yml ps
docker compose -f deploy/milvus/docker-compose.yml stop
docker compose -f deploy/milvus/docker-compose.yml start
```

`stop`, `start`, `restart`, and `down` (without `-v`) keep all data. Deleting `deploy/milvus/volumes/` is the only thing that erases the database; no project command does that. No project command drops or recreates a collection either.

## Migrate the existing FAISS artifacts (no re-embedding)

`src.migrate_faiss_to_milvus` reads a completed artifact directory (`songs.faiss`, `metadata.json`, `manifest.json`), checks that the manifest is a project-generated format-1 build for `all-MiniLM-L6-v2`/384-d and that the FAISS file size matches it, accepts only an exact `IndexFlatIP` (any other index type fails with an explanation), reconstructs the stored vectors in batches, pairs FAISS position *i* with `metadata.json` position *i*, and validates unique IDs, text limits, finite values, and unit norms before writing anything. It then creates the collection (or validates an existing compatible one), performs batched full-record upserts keyed by the existing `song_id`, and verifies the result with Strong consistency: logical count, exact ID set, and every vector and metadata field read back. Only then is the build marked `complete`. Re-running is safe; the same IDs are replaced, never duplicated. The source files are never modified or deleted.

PowerShell:

```powershell
# Validate only (optionally also check the CSV fingerprint and every song_id -> song mapping)
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/sample" --collection songs_sample --csv "data/spotify_millsongdata.csv" --dry-run
# Migrate
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/sample" --collection songs_sample --batch-size 500
# A full artifact, if you have one, goes to its own collection
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/full" --collection songs_full
```

Bash:

```bash
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/sample" --collection songs_sample --csv "data/spotify_millsongdata.csv" --dry-run
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/sample" --collection songs_sample --batch-size 500
python -m src.migrate_faiss_to_milvus --artifacts "artifacts/full" --collection songs_full
```

Options: `--faiss-index`, `--metadata`, `--manifest` override individual source paths; `--batch-size` (1–5000) controls both reconstruction and upsert batches. Exit codes: `0` success, `1` refused (invalid source or incompatible destination; nothing written), `3` Milvus unavailable.

## Build a collection from the CSV

Use this when no completed artifact exists or for a new dataset/model build. `src.build_index` reuses the same preprocessing, token-ID chunking, and pooling code as before; only the destination changed. It checks the destination before the expensive embedding pass, excludes (and records the reason for) any song whose vector is invalid or whose text exceeds a field limit, and imports with the same verification and `importing` → `complete` states as the migration.

PowerShell:

```powershell
# Reproducible 1,000-song development sample (seed 42), into the sample collection
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_sample --sample-size 1000 --seed 42
# Full catalogue, into its own collection (expensive: embeds every valid song)
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_full --full
# Optional: also write a legacy FAISS artifact for baseline verification
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_full --full --faiss-backup "artifacts/full"
```

Bash:

```bash
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_sample --sample-size 1000 --seed 42
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_full --full
python -m src.build_index --csv "data/spotify_millsongdata.csv" --collection songs_full --full --faiss-backup "artifacts/full"
```

`--chunk-size`, `--overlap`, `--batch-size` (model inference), `--upsert-batch-size`, and `--device cpu` are configurable. `--overwrite` only applies to an existing `--faiss-backup` directory.

### Sample and full collections stay separate

- A sample build is refused for a collection named `songs_full` (or containing `full`), and a full build for `songs_sample` (or containing `sample`).
- Each collection has a build manifest in the `song_search_builds` registry collection. Writing into an existing collection requires the same *content signature*: schema version, model identifier and revision, dimension, tokenizer limit, chunking, pooling, dataset fingerprint, build mode, sample seed, and selected song count. Anything else is refused. Use a new collection name (for example `songs_full_v2`) for a different dataset or embedding build.
- An existing collection with a different schema, a non-HNSW or non-COSINE index, or no manifest is refused and left untouched.

## Select the active collection and start the app

The app serves only `MILVUS_COLLECTION`, and only if its manifest says `complete`, the schema/index/manifest validate, the collection is loaded (the app loads it if needed, a non-destructive operation), and the stored count equals the verified count. Switching between sample and full is therefore an explicit configuration change.

PowerShell:

```powershell
$env:MILVUS_COLLECTION = "songs_sample"      # or "songs_full" once built
python -m uvicorn src.app:app --reload
```

Bash:

```bash
MILVUS_COLLECTION=songs_sample python -m uvicorn src.app:app --reload
```

Open <http://127.0.0.1:8000>. `GET /health` performs a live check (connectivity, collection, schema/index, completed manifest, load state, count) and returns `{"ready": true, "indexed_song_count": …, "artifact_mode": "sample"|"full", "collection": …}` or `{"ready": false, "detail": …}`. `POST /search` keeps the same request (`query`, `top_k` 1–50) and response (`rank`, `song_id`, `artist`, `song`, `similarity_score`, `source_link`, `safe_source_url`, `excerpt`). `similarity_score` is the raw cosine similarity, not a percentage or probability. If Milvus or the collection is not ready, `/search` returns 503 with a setup hint; credentials and internal tracebacks are never included. The model and Milvus client are created once at startup, shared by requests, and closed on shutdown. While not ready, the app retries initialization at most every 15 s; once ready it never reconnects or reloads per request.

## Verify migration results

`src.verify_migration` compares Milvus FLAT+COSINE results with the original FAISS `IndexFlatIP` results on the same vectors, same metadata, same queries, and no filters. It uses stored song vectors as queries (no model needed) and, unless `--no-model` is given, the evaluation queries embedded with the original model (development split by default; `--split all` or `final_test` are available because this checks storage equivalence, not relevance). Scores must agree within `--tolerance` (default `1e-5`). Order is only required up to exact ties, and songs tied at the k-th score may differ. Metadata must match `metadata.json`. Milvus reads use Strong consistency.

PowerShell:

```powershell
python -m src.verify_migration --artifacts "artifacts/sample" --collection songs_sample --vector-queries 50 --top-k 10
python -m src.verify_migration --artifacts "artifacts/sample" --collection songs_sample --no-model --vector-queries 100 --report "verify-report.json"
```

Bash:

```bash
python -m src.verify_migration --artifacts "artifacts/sample" --collection songs_sample --vector-queries 50 --top-k 10
python -m src.verify_migration --artifacts "artifacts/sample" --collection songs_sample --no-model --vector-queries 100 --report "verify-report.json"
```

Exit code `0` means every comparison passed. To check persistence, run `docker compose -f deploy/milvus/docker-compose.yml restart`, wait for `/healthz`, and run the verification again.

## Files: runtime versus backup

| Path | Status |
| --- | --- |
| `artifacts/sample/`, `artifacts/smoke/` (`songs.faiss`, `metadata.json`, `manifest.json`) | Kept unchanged as backups and as the verification baseline. **Not read by the app.** |
| `deploy/milvus/volumes/` | The Milvus database (runtime data). Back it up to keep migrated collections. |
| `song_search_builds` collection | Build manifests (state, counts, verification, settings) for each song collection. |
| `faiss-cpu` | Needed only by `migrate_faiss_to_milvus`, `verify_migration`, `build_index --faiss-backup`, and `evaluate_tfidf --artifacts`. |

## What is indexed

The builder validates columns, removes missing/empty lyric rows, normalizes excessive whitespace, and removes only exact duplicates of cleaned `(artist, song, lyrics)`. It keeps same-title songs when artist or lyric text differs. Missing artist/title values use display placeholders. It reports original rows, empty-lyric exclusions, duplicates, valid cleaned count, selected count, and excluded records with reasons in its console output and the collection's build manifest.

There is no training and no song-level train/test split. Embedding a lyric with the frozen pretrained `sentence-transformers/all-MiniLM-L6-v2` model is inference; adding resulting vectors to Milvus is indexing. Every valid song belongs in the searchable collection. The seed-42 sample of about 1,000 valid songs is only a reproducible development pipeline check. Development/final separation applies to evaluation queries, while both sets search the same collection.

Long lyrics are tokenized with the loaded model tokenizer, with no silent whole-lyric truncation. Defaults are 200 content tokens, 32-token overlap, and 168 stride. Special tokens are added after slicing token IDs, and the implementation checks the loaded model's actual input limit. Thus it does not decode a slice and accidentally re-tokenize/truncate it. A 450-token lyric produces `[0:200]`, `[168:368]`, `[336:450]`. (The tokenizer may print "Token indices sequence length is longer than the specified maximum" when a whole lyric is tokenized before slicing; no model input exceeds the limit.)

`200` is an input-token chunk limit. `384` is the output embedding dimension. They measure different things. Each chunk embedding is L2-normalized, normalized chunks for the same song are averaged coordinate-wise, then that mean is L2-normalized. This makes one 384-float vector/song. It can blur emotions that vary across verses, and overlapping text is deliberately represented more than once.

### Milvus collection schema (schema version 1)

| Field | Type | Notes |
| --- | --- | --- |
| `song_id` | `VARCHAR(64)`, primary key, `auto_id=False` | The existing positional song ID string (e.g. `"182"`), unchanged and still a string in the API. |
| `vector` | `FLOAT_VECTOR(384)` | The L2-normalized song vector. |
| `artist` | `VARCHAR(512)` | Required. |
| `song` | `VARCHAR(512)` | Required. |
| `link` | `VARCHAR(2048)`, nullable | Raw source link; the API returns it as `source_link` and derives `safe_source_url`. |
| `lyrics_excerpt` | `VARCHAR(2048)`, nullable | The first 300 characters (plus `…`), shown by the UI as `excerpt`. |

Dynamic fields are disabled. `max_length` is in UTF-8 bytes; the largest values in the cleaned dataset are 44 (artist), 77 (song), 102 (link), and 303 (excerpt) bytes. Over-limit text is never truncated: the migration refuses and the CSV build excludes that song with the reason recorded. The vector index is `index_type="HNSW"`, `metric_type="COSINE"`, `params={"M": 16, "efConstruction": 200}`, with query search parameter `ef=64`. Query vectors are normalized too; COSINE on unit vectors ranks identically to the normalized inner product. Approximate nearest neighbours still do not guarantee that a returned song is a correct mood match.

Raw vector storage is `number_of_songs × 384 × 4 bytes`: for the 57,649 valid cleaned songs that is **88,548,864 bytes** of float32 vectors, excluding model weights, metadata, Milvus/etcd/MinIO overhead, and other application memory. No Milvus memory or latency benchmark has been recorded; do not interpret unrecorded values as benchmarks.

The selected model is principally English-language. It may retrieve non-English lyrics unevenly; this application makes no equal-quality multilingual claim. Lyrics can also be incomplete, metaphorical, incorrectly transcribed, or insufficient to represent the listener's intended mood.

## Evaluation and optional TF-IDF baseline

`evaluation/queries.json` explicitly contains 10 development queries and 20 final-test queries. Use development queries to settle chunking/batching configuration before viewing final test results. To create the fixed human judgment pool:

```powershell
python -m src.evaluate --collection songs_full --export "evaluation/candidates_to_judge.csv" --top-k 10
# A reviewer fills judgment with 0 (not relevant), 1 (partially), or 2 (strongly relevant).
python -m src.evaluate --metrics "evaluation/candidates_to_judge.csv"
python -m src.evaluate --collection songs_full --benchmark
```

(`--collection` defaults to `MILVUS_COLLECTION`; the Bash commands are identical.)

Blank judgment cells remain unjudged; they are never assumed irrelevant. Precision@5 counts only grade `2` as relevant and is reported only for fully judged top five. nDCG@10 requires a fully judged top-ten pool and uses the ideal ordering of that same fixed judged candidate pool for comparisons, rather than redefining an ideal from each method's returned list. Pooled judgments are not exhaustive catalogue ground truth. No labels, relevant songs, quality scores, or measurements are fabricated. Moving storage from FAISS to Milvus does not change which songs are returned (see *Verify migration results*), so it is not a relevance improvement.

`src/tfidf_baseline.py` supplies an optional lexical TF-IDF comparator over the same cleaned song records. It is intentionally separate from, and does not replace, the dense pipeline. Run it on exactly the songs in a collection (or a legacy artifact with `--artifacts`):

```powershell
python -m src.evaluate_tfidf --collection songs_full --csv "data/spotify_millsongdata.csv" --export "evaluation/tfidf_candidates_to_judge.csv" --top-k 10
```

## Tests

Unit tests need neither Docker nor a model download. They use an in-memory fake `MilvusClient` (`tests/fake_milvus.py`) built on real pymilvus schema and index objects.

PowerShell:

```powershell
python -m pytest -q                                   # unit tests; integration tests are skipped
$env:MILVUS_INTEGRATION = "1"; python -m pytest tests/integration -v; Remove-Item Env:MILVUS_INTEGRATION
python -m pytest tests/test_milvus_store.py::test_repeat_import_is_idempotent_without_duplicate_logical_records -q
```

Bash:

```bash
python -m pytest -q
MILVUS_INTEGRATION=1 python -m pytest tests/integration -v
python -m pytest tests/test_milvus_store.py::test_repeat_import_is_idempotent_without_duplicate_logical_records -q
```

Unit tests cover configuration validation, schema and index validation, vector dimension, non-finite, and norm handling, over-limit text, vector-to-song mapping, idempotent repeat imports, stale-record detection, sample/full separation, metadata retrieval, query validation and result limits, response mapping and score ordering, tie-aware FAISS comparison, unsupported FAISS index types, and database-unavailable or collection-not-ready API behaviour. Integration tests (`tests/integration/`) run against the real server configured by `MILVUS_URI`. They create uniquely named `it_songs_*` collections and drop only those. They cover the server-side schema and index, idempotent upserts, Milvus-vs-FAISS parity with exact ties, null metadata, incompatible-collection rejection, and connection failure.
# recommendation-system

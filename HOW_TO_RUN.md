# How to Run the Song Search Engine

This guide walks you through running the entire Lyrics Semantic Song Search Engine using **Docker Compose**.

---

## 1. Prerequisites

- **Docker Desktop** installed and running:
  - On Windows: ensure Docker Desktop is using the **WSL 2 backend**.
  - On Linux / macOS: ensure Docker Engine and Compose v2 are installed.
- **Git** (optional, for cloning).
- No Python installation is strictly required on your host machine if you run inside Docker!

---

## 2. Quick Start (3 Steps)

### Step 1: Start the Services

Open your terminal in this project root directory and start the entire stack:

- **Windows (Command Prompt / PowerShell):**
  ```cmd
  .\run.cmd up
  ```
- **macOS / Linux / Git Bash:**
  ```bash
  chmod +x run.sh
  ./run.sh up
  ```
- **Or standard Docker Compose:**
  ```bash
  docker compose up -d --build
  ```

This starts:
- **Milvus Standalone** (vector database on port `19530`)
- **MinIO** (vector segment storage on ports `9000` & `9001`)
- **etcd** (metadata coordinator)
- **FastAPI Web Application** (search UI on port `8000`)

---

### Step 2: Ingest the Song Collection into Milvus

Before searching songs, populate Milvus with vector embeddings. Run the migration command to import the pre-computed sample:

- **Windows:**
  ```cmd
  .\run.cmd migrate
  ```
- **macOS / Linux / Git Bash:**
  ```bash
  ./run.sh migrate
  ```
- **Or standard Docker Compose:**
  ```bash
  docker compose run --rm migrate
  ```

This takes only a few seconds. It imports the 1,000-song sample vectors from `artifacts/sample/` into the Milvus `songs_sample` collection and registers the build manifest.

---

### Step 3: Open the Search Engine

Open your browser and navigate to:

👉 **[http://localhost:8000](http://localhost:8000)**

You can now type thematic queries like:
- `feeling nostalgic about childhood summer days`
- `broken heart and lonely late nights`
- `celebration, dancing with friends, and good energy`

---

## 3. Daily Development & Useful Commands

| Task | Windows Helper | Bash / Mac Helper | Standard Docker Command |
| :--- | :--- | :--- | :--- |
| **Start everything** | `.\run.cmd up` | `./run.sh up` | `docker compose up -d --build` |
| **Migrate sample data** | `.\run.cmd migrate` | `./run.sh migrate` | `docker compose run --rm migrate` |
| **Build sample from CSV** | `.\run.cmd build` | `./run.sh build` | `docker compose run --rm build-sample` |
| **Follow web logs** | `.\run.cmd logs` | `./run.sh logs` | `docker compose logs -f web` |
| **Check container status**| `.\run.cmd ps` | `./run.sh ps` | `docker compose ps` |
| **Run test suite** | `.\run.cmd test` | `./run.sh test` | `docker compose run --rm test` |
| **Restart services** | `.\run.cmd restart` | `./run.sh restart` | `docker compose restart` |
| **Stop services (keep data)**| `.\run.cmd stop` | `./run.sh stop` | `docker compose stop` |
| **Tear down containers** | `.\run.cmd down` | `./run.sh down` | `docker compose down` |

---

## 4. Live Code Reloading

The `web` container is mounted directly to your host's `./src` and `./static` folders with `--reload` enabled:
- Any changes you make in Python files (`src/*.py`) or front-end templates (`static/*.html`, `static/*.js`) **update instantly** in the running application without needing to rebuild the Docker image!

---

## 5. Service Endpoints & Consoles

When the stack is running:

| Component | URL | Notes |
| :--- | :--- | :--- |
| **FastAPI Search Engine** | [http://localhost:8000](http://localhost:8000) | Web UI and REST API (`/health`, `/search`) |
| **Milvus Health Check** | [http://localhost:9091/healthz](http://localhost:9091/healthz) | Returns `OK` when Milvus is healthy |
| **MinIO Storage Console** | [http://localhost:9001](http://localhost:9001) | Credentials: `minioadmin` / `minioadmin` |
| **Milvus gRPC API** | `localhost:19530` | For Python `pymilvus` client connections |

---

## 6. Advanced Indexing: Building Full Dataset

If you have the full `data/spotify_millsongdata.csv` dataset and want to build the complete collection (~57,000 songs) instead of the 1,000-song sample:

```bash
docker compose run --rm web python -m src.build_index \
  --csv data/spotify_millsongdata.csv \
  --collection songs_full \
  --full
```

*(Note: embedding all 57,000+ songs takes 15–30 minutes on CPU; embeddings are cached into `./deploy/milvus/volumes/`)*.

To switch the web app to serve the full collection, update the `MILVUS_COLLECTION` environment variable in `docker-compose.yml`:
```yaml
      - MILVUS_COLLECTION=songs_full
```
Then restart the web container:
```bash
docker compose restart web
```

---

## 7. Data Persistence & Backup

- All vector collections, metadata, and MinIO segment files persist in:
  `./deploy/milvus/volumes/`
- Running `docker compose stop` or `docker compose down` **will not lose your data**.
- To completely reset and wipe the database, delete the contents of `./deploy/milvus/volumes/` (Milvus will recreate clean directories on the next run).

---

## 8. Hybrid Mode (Developing Locally with Python)

If you prefer editing and running Python directly in your local terminal:
1. Start only the vector database stack:
   ```cmd
   docker compose up -d standalone etcd minio
   ```
2. Activate your local virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Run the uvicorn web server locally:
   ```powershell
   python -m uvicorn src.app:app --reload
   ```
Your local Python application will connect to Milvus at `http://localhost:19530`.

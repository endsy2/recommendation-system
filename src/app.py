"""FastAPI application serving lyrics search from a completed Milvus collection."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .config import ConfigError, MilvusSettings, load_env_file
from .search_engine import ArtifactError, QueryValidationError, SearchEngine

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
# While not ready, retry initialization at most this often; a ready engine is never rebuilt per request.
RETRY_SECONDS = 15.0
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: Annotated[str, Field(max_length=10000)]
    top_k: Annotated[int, Field(default=10, ge=1, le=50)] = 10

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


def _engine_factory(settings: MilvusSettings) -> SearchEngine:
    return SearchEngine.from_milvus(settings, os.environ.get("SONG_SEARCH_DEVICE") or None)


async def _initialize(app: FastAPI) -> None:
    """Connect, validate the collection, and load the model once; record a safe reason on failure."""
    async with app.state.init_lock:
        if app.state.engine is not None or app.state.settings is None:
            return
        app.state.last_attempt = time.monotonic()
        try:
            app.state.engine = await asyncio.to_thread(app.state.engine_factory, app.state.settings)
            app.state.load_error = None
        except ArtifactError as exc:
            logger.warning("Search engine not ready: %s", exc, exc_info=exc.__cause__ is not None)
            app.state.load_error = str(exc)


async def _current_engine(app: FastAPI) -> SearchEngine | None:
    if app.state.engine is None and app.state.settings is not None:
        if time.monotonic() - app.state.last_attempt >= RETRY_SECONDS:
            await _initialize(app)
    return app.state.engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env_file()
    app.state.engine, app.state.load_error, app.state.last_attempt = None, None, 0.0
    app.state.init_lock = asyncio.Lock()
    if not hasattr(app.state, "engine_factory"):
        app.state.engine_factory = _engine_factory
    try:
        app.state.settings = MilvusSettings.from_env()
    except ConfigError as exc:
        app.state.settings, app.state.load_error = None, f"Invalid Milvus configuration: {exc}"
    await _initialize(app)
    try:
        yield
    finally:
        engine, app.state.engine = app.state.engine, None
        if engine is not None:
            await asyncio.to_thread(engine.close)


app = FastAPI(title="Lyrics Semantic Song Search", lifespan=lifespan)


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, object]:
    engine = await _current_engine(app)
    if engine is None:
        return {"ready": False, "detail": app.state.load_error or "Search engine is not loaded"}
    try:
        # Live check: connectivity, collection, schema/index, completed manifest, load state, count.
        manifest = await asyncio.to_thread(engine.store.check_ready)
    except ArtifactError as exc:
        return {"ready": False, "detail": str(exc)}
    return {"ready": True, "indexed_song_count": int(manifest["imported_song_count"]),
            "artifact_mode": manifest["build_mode"], "collection": engine.collection}


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, object]:
    engine = await _current_engine(app)
    if engine is None:
        raise HTTPException(status_code=503, detail=app.state.load_error or "Search engine is not ready")
    try:
        results, _ = await asyncio.to_thread(engine.search_with_metrics, request.query, request.top_k)
    except QueryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ArtifactError as exc:
        logger.warning("Search failed: %s", exc, exc_info=exc.__cause__ is not None)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": request.query, "results": [result.to_dict() for result in results]}


@app.get("/static/{asset:path}")
async def static_file(asset: str) -> FileResponse:
    if asset not in {"styles.css", "app.js"}:
        raise HTTPException(status_code=404)
    return FileResponse(STATIC / asset)

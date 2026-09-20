"""Optional TF-IDF candidate export over exactly the songs in a Milvus collection (or a legacy FAISS artifact)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import MilvusSettings, load_env_file
from .evaluate import load_queries
from .milvus_store import PK_FIELD, MilvusSongStore
from .preprocess import load_and_clean_csv
from .tfidf_baseline import TfidfBaseline


def _songs_from_collection(collection: str | None) -> list[dict[str, Any]]:
    store = MilvusSongStore.open(MilvusSettings.from_env(collection=collection))
    try:
        return [{"song_id": str(row[PK_FIELD]), "artist": row["artist"], "song": row["song"], "source_link": row.get("link")}
                for row in store.iter_metadata()]
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional lexical TF-IDF comparator for the dense evaluation pool.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", help="Milvus collection whose exact songs are compared")
    source.add_argument("--artifacts", help="Legacy FAISS artifact directory whose metadata.json lists the songs")
    parser.add_argument("--csv", required=True, help="Original song CSV used to build the collection")
    parser.add_argument("--queries", default="evaluation/queries.json")
    parser.add_argument("--export", required=True, help="Write blank-judgment candidates for the same review workflow")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        raise SystemExit("--top-k must be from 1 to 50")
    load_env_file()
    if args.collection:
        metadata = _songs_from_collection(args.collection)
    else:
        metadata = json.loads((Path(args.artifacts) / "metadata.json").read_text(encoding="utf-8"))
    wanted_ids = sorted((str(item["song_id"]) for item in metadata), key=int)
    records, _ = load_and_clean_csv(args.csv)
    by_id = {record.song_id: record for record in records}
    missing = [song_id for song_id in wanted_ids if song_id not in by_id]
    if missing:
        raise SystemExit("Song IDs do not align with this CSV after cleaning; use the original build CSV.")
    selected = [by_id[song_id] for song_id in wanted_ids]
    started = time.perf_counter()
    baseline = TfidfBaseline.from_records(selected)
    build_seconds = time.perf_counter() - started
    metadata_by_id = {str(item["song_id"]): item for item in metadata}
    with Path(args.export).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "split", "query", "rank", "song_id", "artist", "song", "similarity_score", "source_link", "judgment"])
        writer.writeheader()
        for query in load_queries(args.queries):
            for rank, (song_id, score) in enumerate(baseline.search(query["query"], args.top_k), start=1):
                item = metadata_by_id[song_id]
                writer.writerow({"query_id": query["id"], "split": query["split"], "query": query["query"], "rank": rank,
                                 "song_id": song_id, "artist": item["artist"], "song": item["song"],
                                 "similarity_score": score, "source_link": item.get("source_link") or "", "judgment": ""})
    print(f"Built TF-IDF on {len(selected)} same songs in {build_seconds:.3f}s; wrote {args.export}")


if __name__ == "__main__":
    main()

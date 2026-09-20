"""Human-judgment export, conservative metrics, and optional timing workflow."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .config import MilvusSettings, load_env_file
from .search_engine import SearchEngine


def load_queries(path: str | Path) -> list[dict[str, str]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(set(("id", "split", "query")) - set(item) for item in value):
        raise ValueError("queries JSON must be a list of {id, split, query} objects")
    return value


def export_candidates(engine: SearchEngine, queries: list[dict[str, str]], output: str | Path, top_k: int = 10) -> None:
    path = Path(output)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "split", "query", "rank", "song_id", "artist", "song", "similarity_score", "source_link", "judgment"])
        writer.writeheader()
        for item in queries:
            for result in engine.search(item["query"], top_k):
                writer.writerow({"query_id": item["id"], "split": item["split"], "query": item["query"], "rank": result.rank,
                                 "song_id": result.song_id, "artist": result.artist, "song": result.song,
                                 "similarity_score": result.similarity_score, "source_link": result.source_link or "", "judgment": ""})


def calculate_metrics(judgments_csv: str | Path) -> dict[str, Any]:
    """Only score a query when its complete top-k pool is judged; blanks are not irrelevance."""
    grouped: dict[str, list[dict[str, str]]] = {}
    with Path(judgments_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["query_id"], []).append(row)
    p5: list[float] = []
    ndcg10: list[float] = []
    incomplete: list[str] = []
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rank"]))
        top5, top10 = rows[:5], rows[:10]
        if len(top5) == 5 and all(row.get("judgment", "").strip() in {"0", "1", "2"} for row in top5):
            p5.append(sum(row["judgment"].strip() == "2" for row in top5) / 5)
        else:
            incomplete.append(query_id)
            continue
        if len(top10) == 10 and all(row.get("judgment", "").strip() in {"0", "1", "2"} for row in top10):
            grades = [int(row["judgment"]) for row in top10]
            dcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
            ideal = sorted(grades, reverse=True)
            idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
            if idcg > 0:
                ndcg10.append(dcg / idcg)
    return {"precision_at_5": sum(p5) / len(p5) if p5 else None, "precision_at_5_query_count": len(p5),
            "ndcg_at_10": sum(ndcg10) / len(ndcg10) if ndcg10 else None, "ndcg_at_10_query_count": len(ndcg10),
            "incomplete_or_unscored_query_ids": sorted(set(incomplete)),
            "policy": "Blank judgments are unjudged, never grade 0. nDCG ideal ranks only the same fixed exported top-10 judged pool."}


def benchmark(engine: SearchEngine, queries: list[dict[str, str]], top_k: int = 10) -> dict[str, float | int]:
    measurements = [engine.search_with_metrics(query["query"], top_k)[1] for query in queries]
    return {"queries": len(measurements), **{f"median_{key}": statistics.median(item[key] for item in measurements)
            for key in ("query_embedding_seconds", "vector_search_seconds", "end_to_end_seconds")}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidates, score completed human judgments, or benchmark dense search.")
    parser.add_argument("--collection", help="Milvus collection to evaluate (default: MILVUS_COLLECTION)")
    parser.add_argument("--queries", default="evaluation/queries.json")
    parser.add_argument("--device", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", help="Write a blank-judgment CSV for manual review")
    group.add_argument("--metrics", help="Read manually completed candidate CSV")
    group.add_argument("--benchmark", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.metrics:
        print(json.dumps(calculate_metrics(args.metrics), indent=2))
        return
    load_env_file()
    engine = SearchEngine.from_milvus(MilvusSettings.from_env(collection=args.collection), args.device)
    try:
        queries = load_queries(args.queries)
        if args.export:
            export_candidates(engine, queries, args.export, args.top_k)
            print(f"Wrote candidates from collection {engine.collection!r} to {args.export}; "
                  "assign only 0, 1, or 2 when reviewed.")
        else:
            print(json.dumps({"collection": engine.collection, **benchmark(engine, queries, args.top_k)}, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()

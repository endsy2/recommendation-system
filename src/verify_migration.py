"""Compare Milvus FLAT+COSINE results against the original FAISS IndexFlatIP baseline.

Both sides use the same stored vectors, the same metadata, the same queries, and no filters. Scores
must agree within a float tolerance; ordering is only required up to exact (within-tolerance) ties.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import MilvusSettings, load_env_file
from .evaluate import load_queries
from .faiss_artifacts import ArtifactExportError, FaissArtifact, load_faiss_artifact
from .milvus_store import MilvusSongStore, NotReadyError, StoredHit

DEFAULT_TOLERANCE = 1e-5


@dataclass
class QueryComparison:
    label: str
    returned: int
    max_score_difference: float
    top1_agrees: bool
    ranking_agrees_up_to_ties: bool
    exact_id_order_matches: bool
    metadata_matches: bool
    problems: list[str]

    @property
    def passed(self) -> bool:
        return not self.problems


def compare_rankings(
    label: str,
    baseline_ids: Sequence[str],
    baseline_scores: Sequence[float],
    candidate_ids: Sequence[str],
    candidate_scores: Sequence[float],
    exact_scores: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> QueryComparison:
    """Tie-aware agreement between two top-k lists.

    ``exact_scores`` maps every candidate ID to its exact baseline similarity with the query, so an ID
    that Milvus returns but FAISS cut off at a tied boundary can still be checked.
    """
    problems: list[str] = []
    if len(candidate_ids) != len(baseline_ids):
        problems.append(f"returned {len(candidate_ids)} results, baseline returned {len(baseline_ids)}")
    if len(set(candidate_ids)) != len(candidate_ids):
        problems.append("candidate results contain duplicate song IDs")
    max_diff = 0.0
    for rank, (candidate_score, baseline_score) in enumerate(zip(candidate_scores, baseline_scores), start=1):
        diff = abs(float(candidate_score) - float(baseline_score))
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            problems.append(f"rank {rank} score {candidate_score:.7f} vs baseline {baseline_score:.7f}")
    for song_id, score in zip(candidate_ids, candidate_scores):
        exact = exact_scores.get(song_id)
        if exact is None:
            problems.append(f"song {song_id} missing from exact baseline scores")
        elif abs(float(score) - exact) > tolerance:
            problems.append(f"song {song_id} score {score:.7f} vs exact baseline {exact:.7f}")
    if any(later > earlier + tolerance for earlier, later in zip(candidate_scores, candidate_scores[1:])):
        problems.append("candidate scores are not in descending order")
    # Songs strictly above the k-th score (beyond tolerance) are not ties and must be present.
    if baseline_scores:
        boundary = float(baseline_scores[-1])
        required = {song_id for song_id, score in zip(baseline_ids, baseline_scores) if score > boundary + tolerance}
        absent = sorted(required - set(candidate_ids))
        if absent:
            problems.append(f"missing non-tied baseline results: {', '.join(absent[:10])}")
    ranking_ok = not problems
    top1_ok = bool(candidate_ids) and bool(baseline_ids) and (
        candidate_ids[0] == baseline_ids[0] or abs(float(candidate_scores[0]) - float(baseline_scores[0])) <= tolerance
        and abs(exact_scores.get(candidate_ids[0], float("inf")) - float(baseline_scores[0])) <= tolerance
    )
    if baseline_ids and not top1_ok:
        problems.append(f"top result {candidate_ids[0] if candidate_ids else None} differs from {baseline_ids[0]}")
    return QueryComparison(label, len(candidate_ids), max_diff, top1_ok, ranking_ok,
                           list(candidate_ids) == list(baseline_ids), True, problems)


def _metadata_problems(hits: Sequence[StoredHit], metadata_by_id: dict[str, dict[str, Any]]) -> list[str]:
    problems = []
    for hit in hits:
        item = metadata_by_id.get(hit.song_id)
        if item is None:
            problems.append(f"song {hit.song_id} not in artifact metadata")
            continue
        stored = {"artist": hit.artist, "song": hit.song, "source_link": hit.link, "excerpt": hit.lyrics_excerpt}
        for key, value in stored.items():
            if item.get(key) != value:
                problems.append(f"song {hit.song_id} {key} differs from artifact metadata")
    return problems


def compare_query(label: str, vector: np.ndarray, artifact: FaissArtifact, all_vectors: np.ndarray, store: MilvusSongStore,
                  top_k: int, tolerance: float, metadata_by_id: dict[str, dict[str, Any]]) -> QueryComparison:
    query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
    limit = min(top_k, artifact.count)
    scores, positions = artifact.index.search(query, limit)
    baseline_ids = [artifact.song_ids[int(position)] for position in positions[0] if position >= 0]
    baseline_scores = [float(score) for score, position in zip(scores[0], positions[0]) if position >= 0]
    hits = store.search(vector, top_k, consistency_level="Strong")
    position_by_id = {song_id: position for position, song_id in enumerate(artifact.song_ids)}
    exact = {hit.song_id: float(all_vectors[position_by_id[hit.song_id]] @ query[0])
             for hit in hits if hit.song_id in position_by_id}
    result = compare_rankings(label, baseline_ids, baseline_scores, [hit.song_id for hit in hits],
                              [hit.score for hit in hits], exact, tolerance)
    metadata_problems = _metadata_problems(hits, metadata_by_id)
    if metadata_problems:
        result.metadata_matches = False
        result.problems.extend(metadata_problems)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_faiss_artifact(args.artifacts)
    all_vectors = np.vstack([block for _, block in artifact.iter_vector_batches(1000)])
    metadata_by_id = {str(item["song_id"]): item for item in artifact.metadata}
    store = MilvusSongStore.open(MilvusSettings.from_env(collection=args.collection))
    try:
        if store.song_count != artifact.count:
            raise NotReadyError(f"Collection holds {store.song_count} songs; artifact holds {artifact.count}")
        queries: list[tuple[str, np.ndarray]] = []
        # Stored song vectors as queries: needs no model, exercises exact self-matches and near ties.
        if args.vector_queries > 0:
            step = max(1, artifact.count // args.vector_queries)
            for position in list(range(0, artifact.count, step))[: args.vector_queries]:
                queries.append((f"song-vector:{artifact.song_ids[position]}", all_vectors[position]))
        if not args.no_model:
            from .embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(artifact.manifest["model"]["identifier"], device=args.device)
            for item in load_queries(args.queries):
                if args.split == "all" or item["split"] == args.split:
                    queries.append((f"{item['id']}:{item['query']}", np.asarray(embedder.encode_query(item["query"]), np.float32)))
        comparisons = [compare_query(label, vector, artifact, all_vectors, store, args.top_k, args.tolerance, metadata_by_id)
                       for label, vector in queries]
    finally:
        store.close()
    failed = [item for item in comparisons if not item.passed]
    return {
        "collection": args.collection or MilvusSettings.from_env().collection,
        "artifact": str(artifact.directory),
        "songs": artifact.count,
        "queries_compared": len(comparisons),
        "top_k": args.top_k,
        "score_tolerance": args.tolerance,
        "max_score_difference": max((item.max_score_difference for item in comparisons), default=0.0),
        "top1_agreement": sum(item.top1_agrees for item in comparisons),
        "ranking_agreement_up_to_ties": sum(item.ranking_agrees_up_to_ties for item in comparisons),
        "exact_id_order_matches": sum(item.exact_id_order_matches for item in comparisons),
        "metadata_matches": sum(item.metadata_matches for item in comparisons),
        "passed": not failed,
        "failures": [asdict(item) for item in failed[:20]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Milvus FLAT+COSINE results against the FAISS IndexFlatIP baseline.")
    parser.add_argument("--artifacts", required=True, help="Original FAISS artifact directory (read-only)")
    parser.add_argument("--collection", help="Milvus collection migrated from it (default: MILVUS_COLLECTION)")
    parser.add_argument("--queries", default="evaluation/queries.json")
    parser.add_argument("--split", choices=("all", "development", "final_test"), default="development",
                        help="Evaluation queries to embed (default: development only)")
    parser.add_argument("--vector-queries", type=int, default=25, help="Stored song vectors to use as extra queries")
    parser.add_argument("--no-model", action="store_true", help="Skip text queries (no model load)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", help="Optional path for the JSON report")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k must be from 1 to 50")
    load_env_file()
    try:
        report = run(args)
    except (ArtifactExportError, NotReadyError) as exc:
        print(f"Verification could not run: {exc}", file=sys.stderr)
        return 3
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

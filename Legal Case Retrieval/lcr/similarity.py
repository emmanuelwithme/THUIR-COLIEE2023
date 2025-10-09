from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import EmbeddingsData

SimilarityMetric = Literal["dot", "cos"]


def score_queries(
    query_ids: Sequence[str],
    query_embeddings: EmbeddingsData,
    candidate_embeddings: EmbeddingsData,
    *,
    metric: SimilarityMetric = "dot",
) -> Tuple[List[str], torch.Tensor | None, List[str]]:
    """
    Compute similarity scores for the requested query IDs.

    Returns:
        ordered_query_ids: List of query ids with available embeddings.
        score_matrix: Tensor of shape (len(ordered_query_ids), num_candidates) or ``None`` when empty.
        missing: Query ids without embeddings.
    """
    filtered_queries, missing = query_embeddings.slice_by_ids(query_ids)
    if len(filtered_queries) == 0:
        return [], None, missing

    candidate_matrix = candidate_embeddings.embeddings
    query_matrix = filtered_queries.embeddings

    if metric == "dot":
        score_matrix = torch.matmul(query_matrix, candidate_matrix.T)
    elif metric == "cos":
        qnorm = F.normalize(query_matrix, dim=-1)
        cnorm = F.normalize(candidate_matrix, dim=-1)
        score_matrix = torch.matmul(qnorm, cnorm.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return filtered_queries.ids, score_matrix, missing


def rank_candidates(
    query_ids: Sequence[str],
    query_embeddings: EmbeddingsData,
    candidate_embeddings: EmbeddingsData,
    *,
    metric: SimilarityMetric = "dot",
    run_tag: str | None = None,
    topk: int | None = None,
) -> Tuple[List[str], List[str]]:
    ordered_query_ids, score_matrix, missing = score_queries(
        query_ids, query_embeddings, candidate_embeddings, metric=metric
    )
    if not ordered_query_ids or score_matrix is None:
        return [], missing

    run_name = run_tag or f"{metric}_run"
    lines: List[str] = []
    candidate_ids = candidate_embeddings.ids

    # torch.sort with stable flag preserves candidate order for equal scores
    sorted_scores, sorted_indices = torch.sort(
        score_matrix, dim=1, descending=True, stable=True
    )

    if topk is not None:
        sorted_scores = sorted_scores[:, :topk]
        sorted_indices = sorted_indices[:, :topk]

    sorted_scores = sorted_scores.cpu()
    sorted_indices = sorted_indices.cpu()

    for row, qid in enumerate(ordered_query_ids):
        for rank, (idx, score) in enumerate(
            zip(sorted_indices[row].tolist(), sorted_scores[row].tolist()), start=1
        ):
            doc_id = candidate_ids[idx]
            lines.append(f"{qid} Q0 {doc_id} {rank} {score} {run_name}")

    return lines, missing


def compute_similarity_and_save(
    query_ids: Sequence[str],
    query_embeddings: EmbeddingsData,
    candidate_embeddings: EmbeddingsData,
    output_path: str | Path,
    *,
    metric: SimilarityMetric = "dot",
    run_tag: str | None = None,
    topk: int | None = None,
) -> List[str]:
    """
    Convenience wrapper that writes TREC-formatted rankings to `output_path`
    and returns missing query IDs.
    """
    lines, missing = rank_candidates(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        metric=metric,
        run_tag=run_tag,
        topk=topk,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return missing

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import (
    EmbeddingsData,
    normalize_case_id,
    resolve_query_candidate_scope,
)

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
    query_to_candidate_ids: Mapping[str, Sequence[str]] | None = None,
    fallback_to_all_candidates_if_scope_missing: bool = False,
) -> Tuple[List[str], List[str]]:
    lines, _scores, missing = rank_candidates_with_scores(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        metric=metric,
        run_tag=run_tag,
        topk=topk,
        query_to_candidate_ids=query_to_candidate_ids,
        fallback_to_all_candidates_if_scope_missing=fallback_to_all_candidates_if_scope_missing,
    )
    return lines, missing


def rank_candidates_with_scores(
    query_ids: Sequence[str],
    query_embeddings: EmbeddingsData,
    candidate_embeddings: EmbeddingsData,
    *,
    metric: SimilarityMetric = "dot",
    run_tag: str | None = None,
    topk: int | None = None,
    query_to_candidate_ids: Mapping[str, Sequence[str]] | None = None,
    fallback_to_all_candidates_if_scope_missing: bool = False,
) -> Tuple[List[str], Dict[str, Dict[str, float]], List[str]]:
    """
    Rank candidates for each query and return both TREC lines and score dict.

    When `query_to_candidate_ids` is provided, each query only scores within
    its scoped candidate subset.
    """
    run_name = run_tag or f"{metric}_run"
    lines: List[str] = []
    scores_by_query: Dict[str, Dict[str, float]] = {}

    if query_to_candidate_ids is None:
        ordered_query_ids, score_matrix, missing = score_queries(
            query_ids, query_embeddings, candidate_embeddings, metric=metric
        )
        if not ordered_query_ids or score_matrix is None:
            return [], {}, missing

        candidate_ids = candidate_embeddings.ids
        sorted_scores, sorted_indices = torch.sort(
            score_matrix, dim=1, descending=True, stable=True
        )
        if topk is not None:
            sorted_scores = sorted_scores[:, :topk]
            sorted_indices = sorted_indices[:, :topk]
        sorted_scores = sorted_scores.cpu()
        sorted_indices = sorted_indices.cpu()

        for row, qid in enumerate(ordered_query_ids):
            row_scores: Dict[str, float] = {}
            for rank, (idx, score) in enumerate(
                zip(sorted_indices[row].tolist(), sorted_scores[row].tolist()), start=1
            ):
                doc_id = candidate_ids[idx]
                lines.append(f"{qid} Q0 {doc_id} {rank} {score} {run_name}")
                row_scores[doc_id] = float(score)
            scores_by_query[qid] = row_scores
        return lines, scores_by_query, missing

    filtered_queries, missing = query_embeddings.slice_by_ids(query_ids)
    if len(filtered_queries) == 0:
        return [], {}, missing

    candidate_ids = candidate_embeddings.ids
    candidate_matrix = candidate_embeddings.embeddings
    candidate_id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    all_candidate_ids = list(candidate_ids)
    all_candidate_indices = list(range(len(candidate_ids)))

    normalized_candidate_matrix = None
    if metric == "cos":
        normalized_candidate_matrix = F.normalize(candidate_matrix, dim=-1)
    elif metric != "dot":
        raise ValueError(f"Unsupported metric: {metric}")

    for row_idx, qid in enumerate(filtered_queries.ids):
        scoped_candidates = query_to_candidate_ids.get(qid)
        if scoped_candidates is None:
            scoped_candidates = query_to_candidate_ids.get(f"{qid}.txt")
        if scoped_candidates is None:
            if fallback_to_all_candidates_if_scope_missing:
                selected_ids = all_candidate_ids
                selected_indices = all_candidate_indices
            else:
                scores_by_query[qid] = {}
                continue
        else:
            seen: set[str] = set()
            selected_ids: List[str] = []
            selected_indices: List[int] = []
            for raw_doc_id in scoped_candidates:
                doc_id = normalize_case_id(raw_doc_id)
                if doc_id in seen:
                    continue
                idx = candidate_id_to_index.get(doc_id)
                if idx is None:
                    continue
                seen.add(doc_id)
                selected_ids.append(doc_id)
                selected_indices.append(idx)

            if not selected_ids:
                scores_by_query[qid] = {}
                continue

        index_tensor = torch.tensor(
            selected_indices, dtype=torch.long, device=candidate_matrix.device
        )
        query_vec = filtered_queries.embeddings[row_idx]
        if metric == "dot":
            selected_matrix = candidate_matrix.index_select(0, index_tensor)
            score_vec = torch.matmul(selected_matrix, query_vec)
        else:
            selected_matrix = normalized_candidate_matrix.index_select(0, index_tensor)
            normalized_query = F.normalize(query_vec.unsqueeze(0), dim=-1).squeeze(0)
            score_vec = torch.matmul(selected_matrix, normalized_query)

        sorted_scores, sorted_indices = torch.sort(score_vec, descending=True, stable=True)
        if topk is not None:
            sorted_scores = sorted_scores[:topk]
            sorted_indices = sorted_indices[:topk]

        row_scores: Dict[str, float] = {}
        for rank, (subset_idx, score) in enumerate(
            zip(sorted_indices.cpu().tolist(), sorted_scores.cpu().tolist()), start=1
        ):
            doc_id = selected_ids[subset_idx]
            lines.append(f"{qid} Q0 {doc_id} {rank} {score} {run_name}")
            row_scores[doc_id] = float(score)
        scores_by_query[qid] = row_scores

    return lines, scores_by_query, missing


def compute_similarity_and_save(
    query_ids: Sequence[str],
    query_embeddings: EmbeddingsData,
    candidate_embeddings: EmbeddingsData,
    output_path: str | Path,
    *,
    metric: SimilarityMetric = "dot",
    run_tag: str | None = None,
    topk: int | None = None,
    query_to_candidate_ids: Mapping[str, Sequence[str]] | None = None,
    query_candidate_scope_path: str | Path | None = None,
    fallback_to_all_candidates_if_scope_missing: bool = False,
) -> List[str]:
    """
    Convenience wrapper that writes TREC-formatted rankings to `output_path`
    and returns missing query IDs.
    """
    resolved_scope, _scope_source = resolve_query_candidate_scope(
        query_to_candidate_ids=query_to_candidate_ids,
        query_candidate_scope_path=query_candidate_scope_path,
    )

    lines, missing = rank_candidates(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        metric=metric,
        run_tag=run_tag,
        topk=topk,
        query_to_candidate_ids=resolved_scope,
        fallback_to_all_candidates_if_scope_missing=fallback_to_all_candidates_if_scope_missing,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return missing

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data import EmbeddingsData
from .embeddings import generate_embeddings
from .similarity import score_queries


@dataclass
class SimilarityArtifacts:
    scores: Dict[str, Dict[str, float]]
    trec_path: Path
    candidate_ids: List[str]
    query_ids: List[str]
    missing_queries: List[str]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def query_count(self) -> int:
        return len(self.query_ids)


def generate_similarity_artifacts(
    model,
    tokenizer,
    device: torch.device,
    *,
    candidate_dir: str | Path,
    query_dir: str | Path,
    query_ids: Sequence[str],
    trec_output_path: str | Path,
    run_tag: str,
    batch_size: int = 1,
    max_length: int = 4096,
    quick_test: bool = False,
    candidate_files_override: Optional[Sequence[str]] = None,
    candidate_limit: int = 20,
    query_limit: int = 5,
    verbose: bool = True,
) -> SimilarityArtifacts:
    """
    Produce embeddings for the provided candidates and queries, compute
    dot-product similarities, and persist TREC-formatted rankings.
    """
    candidate_dir = Path(candidate_dir)
    query_dir = Path(query_dir)
    trec_output_path = Path(trec_output_path)

    if quick_test and candidate_files_override:
        candidate_files = [
            candidate_dir / fname for fname in candidate_files_override
            if (candidate_dir / fname).is_file()
        ]
    else:
        candidate_files = sorted(candidate_dir.glob("*.txt"))
        if quick_test and candidate_files:
            k = min(candidate_limit, len(candidate_files))
            candidate_files = random.sample(candidate_files, k)
    candidate_files = sorted(candidate_files)

    candidate_ids = [path.stem for path in candidate_files]
    candidate_texts = [path.read_text(encoding="utf-8").strip() for path in candidate_files]

    incoming_qids = list(query_ids)
    if quick_test and len(incoming_qids) > query_limit:
        incoming_qids = random.sample(incoming_qids, query_limit)

    query_texts: List[str] = []
    actual_query_ids: List[str] = []
    missing_files: List[str] = []

    for qid in incoming_qids:
        qid_str = str(qid).split(".")[0]
        candidates = [query_dir / f"{qid_str}.txt", query_dir / f"{qid_str.zfill(6)}.txt"]
        actual_path = next((path for path in candidates if path.exists()), None)
        if actual_path:
            query_texts.append(actual_path.read_text(encoding="utf-8").strip())
            actual_query_ids.append(qid_str)
        else:
            missing_files.append(qid_str)

    if verbose:
        print(f"🔹 Queries found: {len(actual_query_ids)}/{len(incoming_qids)} in {query_dir}")
        if not actual_query_ids and incoming_qids:
            print(f"⚠️ None of the query files were found. Example missing IDs: {missing_files[:5]}")
        print("🔹 Generating candidate embeddings...")

    def encode_batch(inputs):
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return model.encode(inputs)
        return model.encode(inputs)

    candidate_embeddings = generate_embeddings(
        candidate_texts,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=verbose,
        progress_desc="Candidate embeddings",
    )
    if verbose:
        print("🔹 Generating query embeddings...")
    query_embeddings = generate_embeddings(
        query_texts,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=verbose,
        progress_desc="Query embeddings",
    )

    if device.type == "cuda":
        candidate_embeddings = candidate_embeddings.to(device)
        query_embeddings = query_embeddings.to(device)

    candidate_data = EmbeddingsData(candidate_ids, candidate_embeddings)
    query_data = EmbeddingsData(actual_query_ids, query_embeddings)

    ordered_qids, score_matrix, missing_from_scores = score_queries(
        actual_query_ids,
        query_data,
        candidate_data,
        metric="dot",
    )

    combined_missing = sorted(set(missing_files + missing_from_scores))
    if verbose and combined_missing:
        print(f"⚠️ Missing embeddings for {len(combined_missing)} queries: {combined_missing}")

    lines: List[str] = []
    scores_dict: Dict[str, Dict[str, float]] = {}
    if ordered_qids and score_matrix is not None:
        sorted_scores, sorted_indices = torch.sort(
            score_matrix, dim=1, descending=True, stable=True
        )
        candidate_ids = candidate_data.ids
        sorted_scores = sorted_scores.cpu()
        sorted_indices = sorted_indices.cpu()

        for row, qid in enumerate(ordered_qids):
            row_scores: Dict[str, float] = {}
            for rank, (idx, score) in enumerate(
                zip(sorted_indices[row].tolist(), sorted_scores[row].tolist()), start=1
            ):
                doc_id = candidate_ids[idx]
                lines.append(f"{qid} Q0 {doc_id} {rank} {score} {run_tag}")
                row_scores[doc_id] = float(score)
            scores_dict[qid] = row_scores

    trec_output_path.parent.mkdir(parents=True, exist_ok=True)
    trec_output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if verbose:
        print(f"✅ Saved similarity scores to {trec_output_path}")
        if quick_test:
            print(
                f"[QUICK_TEST] Using {len(candidate_ids)} candidates and {len(actual_query_ids)} queries"
            )

    return SimilarityArtifacts(
        scores=scores_dict,
        trec_path=trec_output_path,
        candidate_ids=candidate_ids,
        query_ids=actual_query_ids,
        missing_queries=combined_missing,
    )

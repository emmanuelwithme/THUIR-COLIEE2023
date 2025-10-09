from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def record_result(
    model_name: str,
    topk: int,
    trec_file: str,
    f1: float,
    precision: float,
    recall: float,
    notes: str = "",
    csv_path: str | Path = "./Legal Case Retrieval/results/experiment_results.csv",
) -> None:
    """
    Append a single experiment run to the consolidated CSV log.  The caller is
    responsible for keeping the values consistent across runs (e.g. unit of
    `topk` or path style for `trec_file`).
    """
    columns = ["model_name", "topk", "trec_file", "f1", "precision", "recall", "notes"]
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)

    new_row: Dict[str, Any] = {
        "model_name": model_name,
        "topk": topk,
        "trec_file": trec_file,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch


class EmbeddingsData:
    """
    Container holding a set of document identifiers and their corresponding vectors.

    Besides providing convenient (de-)serialisation helpers this class keeps a
    dictionary view (`id2vec`) so lookups are O(1).  The constructor accepts any
    tensor-like input for `embeddings` and will convert it into a contiguous
    `torch.Tensor`.
    """

    def __init__(self, ids: Sequence[str], embeddings: torch.Tensor):
        self.ids = [str(i) for i in ids]
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2-D tensor")
        if embeddings.shape[0] != len(self.ids):
            raise ValueError(
                "Number of embeddings does not match number of ids "
                f"({embeddings.shape[0]} vs {len(self.ids)})"
            )
        self.embeddings = embeddings
        self.id2vec = {idx: vec for idx, vec in zip(self.ids, self.embeddings)}

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item: str) -> bool:
        return item in self.id2vec

    def save(self, path: str | Path) -> None:
        """
        Serialise the current instance to a pickle file.  The directory will be
        created automatically if it does not exist yet.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self.ids, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingsData":
        """
        Load an `EmbeddingsData` instance that was previously saved via
        :meth:`save`.
        """
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["ids"], data["embeddings"])

    def slice_by_ids(self, ids: Iterable[str]) -> Tuple["EmbeddingsData", List[str]]:
        """
        Create a new `EmbeddingsData` instance that only contains vectors for
        the requested ids.  Missing ids are returned alongside the filtered
        instance.
        """
        selected_ids: List[str] = []
        selected_vecs = []
        missing: List[str] = []

        for raw_idx in ids:
            idx = str(raw_idx)
            vec = self.id2vec.get(idx)
            if vec is None:
                missing.append(idx)
                continue
            selected_ids.append(idx)
            selected_vecs.append(vec)

        if selected_vecs:
            stacked = torch.stack(selected_vecs)
        else:
            embedding_dim = self.embeddings.shape[1]
            stacked = torch.empty((0, embedding_dim), dtype=self.embeddings.dtype)

        return EmbeddingsData(selected_ids, stacked), missing


def load_query_ids(path: str | Path, *, limit: int | None = None) -> List[str]:
    """
    Load query identifiers (one per line).  Empty lines are ignored.  A
    positive `limit` keeps only the first `limit` entries.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if limit is not None and limit >= 0:
        ids = ids[:limit]
    return ids


def read_text_directory(
    directory: str | Path,
    *,
    suffix: str = ".txt",
    encoding: str = "utf-8",
    sort: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Read all files with `suffix` from `directory` and return a pair containing
    the document identifiers (file stems) and their text contents.
    """
    directory = Path(directory)
    files = [p for p in directory.glob(f"*{suffix}") if p.is_file()]
    if sort:
        files.sort()

    ids: List[str] = []
    texts: List[str] = []
    for file_path in files:
        ids.append(file_path.stem)
        texts.append(file_path.read_text(encoding=encoding).strip())
    return ids, texts

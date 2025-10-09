from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import torch
from tqdm import tqdm

from .data import EmbeddingsData, read_text_directory

Batch = Dict[str, torch.Tensor]
EncodeFn = Callable[[Batch], torch.Tensor]


def _prepare_tokenizer_kwargs(
    max_length: int | None,
    tokenizer_kwargs: dict | None,
) -> dict:
    kwargs = dict(tokenizer_kwargs or {})
    kwargs.setdefault("return_tensors", "pt")
    kwargs.setdefault("padding", True)
    kwargs.setdefault("truncation", True)
    if max_length is not None:
        kwargs.setdefault("max_length", max_length)
    return kwargs


def generate_embeddings(
    texts: Sequence[str],
    tokenizer,
    *,
    encode_batch: EncodeFn,
    batch_size: int = 8,
    max_length: int | None = 512,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
    tokenizer_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Encode a sequence of texts by repeatedly applying `encode_batch`.

    Args:
        texts: Text samples to encode.
        tokenizer: Tokenizer compatible with the provided `encode_batch`.
        encode_batch: Callable receiving a tokenised batch and returning a
            tensor of embeddings.
        batch_size: Number of texts per batch.
        max_length: Max sequence length to hand to the tokenizer.  Set to
            `None` to rely on tokenizer defaults.
        device: Target computation device.  When omitted we stay on CPU.
        show_progress: Display a progress-bar while encoding.
        progress_desc: Optional custom description for the progress-bar.
        tokenizer_kwargs: Extra keyword arguments forwarded to the tokenizer.
    """
    if len(texts) == 0:
        return torch.empty((0, 0))

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    tokeniser_kwargs = _prepare_tokenizer_kwargs(max_length, tokenizer_kwargs)

    iterator: Iterable[int] = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc or "encoding", total=(len(texts) + batch_size - 1) // batch_size)

    chunks = []
    for start in iterator:
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(batch_texts, **tokeniser_kwargs)
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        with torch.no_grad():
            batch_embeddings = encode_batch(inputs)

        if not isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = torch.as_tensor(batch_embeddings)

        chunks.append(batch_embeddings.detach().cpu())

    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))


def generate_embeddings_for_directory(
    directory: str | Path,
    tokenizer,
    *,
    encode_batch: EncodeFn,
    batch_size: int = 8,
    max_length: int | None = 512,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    tokenizer_kwargs: dict | None = None,
    suffix: str = ".txt",
    encoding: str = "utf-8",
) -> EmbeddingsData:
    """
    Convenience wrapper combining :func:`read_text_directory` and
    :func:`generate_embeddings`.
    """
    ids, texts = read_text_directory(directory, suffix=suffix, encoding=encoding)
    if not ids:
        return EmbeddingsData([], torch.empty((0, 0)))

    embeddings = generate_embeddings(
        texts,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=show_progress,
        progress_desc=f"Encoding {Path(directory).name}",
        tokenizer_kwargs=tokenizer_kwargs,
    )
    return EmbeddingsData(ids, embeddings)


def process_directory_to_embeddings(
    directory: str | Path,
    output_path: str | Path,
    tokenizer,
    *,
    encode_batch: EncodeFn,
    batch_size: int = 8,
    max_length: int | None = 512,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    tokenizer_kwargs: dict | None = None,
    suffix: str = ".txt",
    encoding: str = "utf-8",
) -> EmbeddingsData:
    """
    High-level helper for the common pattern of reading a directory, encoding
    its documents, and persisting the resulting embeddings in a pickle file.
    """
    data = generate_embeddings_for_directory(
        directory,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=show_progress,
        tokenizer_kwargs=tokenizer_kwargs,
        suffix=suffix,
        encoding=encoding,
    )
    data.save(output_path)
    return data

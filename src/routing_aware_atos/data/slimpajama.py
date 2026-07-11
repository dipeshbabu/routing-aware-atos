from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PackedSequence:
    """One fixed-width model sequence and its reproducibility metadata."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    split: str
    sequence_index: int
    source_split: str

    @property
    def valid_tokens(self) -> int:
        return int(self.attention_mask.sum())


def iter_document_token_ids(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    text_field: str = "text",
    eos_token_id: int | None = None,
    max_tokens: int | None = None,
) -> Iterator[list[int]]:
    """Tokenize streamed documents without loading the dataset into memory."""

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for row in rows:
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_tokens is not None,
            max_length=max_tokens,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        token_ids = [int(token_id) for token_id in token_ids]
        if eos_token_id is not None:
            token_ids.append(int(eos_token_id))
        if token_ids:
            yield token_ids


def pack_token_budget(
    documents: Iterator[Sequence[int]],
    *,
    token_budget: int,
    sequence_length: int,
    bos_token_id: int,
    pad_token_id: int,
    split: str,
    source_split: str,
    start_sequence_index: int = 0,
) -> list[PackedSequence]:
    """Pack exactly ``token_budget`` valid tokens into deterministic sequences.

    Each sequence begins with BOS. Document boundaries are represented by EOS
    tokens supplied by :func:`iter_document_token_ids`. The final sequence may
    be padded, but its padding never contributes to the paper token budget.
    """

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    buffered: deque[int] = deque()
    packed: list[PackedSequence] = []
    remaining = int(token_budget)
    sequence_index = int(start_sequence_index)

    while remaining > 0:
        valid_length = min(sequence_length, remaining)
        content_length = valid_length - 1
        while len(buffered) < content_length:
            try:
                buffered.extend(int(token_id) for token_id in next(documents))
            except StopIteration as exc:
                raise RuntimeError(
                    f"Dataset stream ended with {remaining} tokens left in split {split!r}"
                ) from exc

        tokens = [int(bos_token_id)]
        tokens.extend(buffered.popleft() for _ in range(content_length))
        mask = [1] * valid_length
        if valid_length < sequence_length:
            padding = sequence_length - valid_length
            tokens.extend([int(pad_token_id)] * padding)
            mask.extend([0] * padding)

        packed.append(
            PackedSequence(
                input_ids=np.asarray(tokens, dtype=np.int32),
                attention_mask=np.asarray(mask, dtype=np.uint8),
                split=split,
                sequence_index=sequence_index,
                source_split=source_split,
            )
        )
        sequence_index += 1
        remaining -= valid_length

    return packed


def collect_document_token_budget(
    documents: Iterator[Sequence[int]],
    *,
    token_budget: int,
    sequence_length: int,
    bos_token_id: int,
    pad_token_id: int,
    split: str,
    source_split: str,
    start_sequence_index: int = 0,
) -> list[PackedSequence]:
    """Collect truncated documents without introducing cross-document context."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    sequences: list[PackedSequence] = []
    remaining = int(token_budget)
    sequence_index = int(start_sequence_index)
    while remaining > 0:
        try:
            document = next(documents)
        except StopIteration as exc:
            raise RuntimeError(
                f"Dataset stream ended with {remaining} tokens left in split {split!r}"
            ) from exc
        valid_tokens = [int(bos_token_id), *[int(token_id) for token_id in document]]
        valid_tokens = valid_tokens[: min(sequence_length, remaining)]
        if not valid_tokens:
            continue
        valid_length = len(valid_tokens)
        padding = sequence_length - valid_length
        sequences.append(
            PackedSequence(
                input_ids=np.asarray(valid_tokens + [int(pad_token_id)] * padding, dtype=np.int32),
                attention_mask=np.asarray([1] * valid_length + [0] * padding, dtype=np.uint8),
                split=split,
                sequence_index=sequence_index,
                source_split=source_split,
            )
        )
        sequence_index += 1
        remaining -= valid_length
    return sequences


def collect_full_document_sequences(
    documents: Iterator[Sequence[int]],
    *,
    num_sequences: int,
    sequence_length: int,
    bos_token_id: int,
    split: str,
    source_split: str,
    start_sequence_index: int = 0,
) -> list[PackedSequence]:
    """Collect full-length documents for causal perplexity evaluation."""

    if num_sequences <= 0:
        return []
    sequences: list[PackedSequence] = []
    while len(sequences) < num_sequences:
        try:
            document = next(documents)
        except StopIteration as exc:
            raise RuntimeError(
                f"Dataset stream ended after {len(sequences)} full causal sequences"
            ) from exc
        tokens = [int(bos_token_id), *[int(token_id) for token_id in document]]
        if len(tokens) < sequence_length:
            continue
        sequence_index = start_sequence_index + len(sequences)
        sequences.append(
            PackedSequence(
                input_ids=np.asarray(tokens[:sequence_length], dtype=np.int32),
                attention_mask=np.ones(sequence_length, dtype=np.uint8),
                split=split,
                sequence_index=sequence_index,
                source_split=source_split,
            )
        )
    return sequences


def validate_split_fractions(fractions: Mapping[str, float]) -> None:
    required = {"train", "validation", "test"}
    if set(fractions) != required:
        raise ValueError(f"split_fractions must contain exactly {sorted(required)}")
    values = [float(fractions[name]) for name in ("train", "validation", "test")]
    if any(value <= 0 for value in values):
        raise ValueError("All split fractions must be positive")
    if not np.isclose(sum(values), 1.0, atol=1e-8):
        raise ValueError(f"split fractions must sum to 1.0, got {sum(values)}")


def split_token_budgets(
    total_tokens: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    """Return exact 60/20/20-style token budgets with no rounding loss."""

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    validate_split_fractions(fractions)
    train = int(total_tokens * float(fractions["train"]))
    validation = int(total_tokens * float(fractions["validation"]))
    test = int(total_tokens) - train - validation
    return {"train": train, "validation": validation, "test": test}


def split_counts(sequences: Iterable[PackedSequence]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for sequence in sequences:
        row = counts.setdefault(sequence.split, {"sequences": 0, "tokens": 0})
        row["sequences"] += 1
        row["tokens"] += sequence.valid_tokens
    return counts


__all__ = [
    "PackedSequence",
    "collect_document_token_budget",
    "collect_full_document_sequences",
    "iter_document_token_ids",
    "pack_token_budget",
    "split_counts",
    "split_token_budgets",
    "validate_split_fractions",
]

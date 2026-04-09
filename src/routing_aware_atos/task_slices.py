from __future__ import annotations

from typing import Callable

import numpy as np


def _to_token_strings(tokenizer, input_ids: np.ndarray) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return list(tokenizer.convert_ids_to_tokens(input_ids.tolist()))
    return [str(x) for x in input_ids.tolist()]


def is_repeated_token_sequence(tokenizer, input_ids: np.ndarray) -> bool:
    toks = _to_token_strings(tokenizer, input_ids)
    seen = set()
    for tok in toks:
        if tok in seen:
            return True
        seen.add(tok)
    return False


def is_delimiter_heavy_sequence(tokenizer, input_ids: np.ndarray) -> bool:
    toks = _to_token_strings(tokenizer, input_ids)
    delimiters = {"(", ")", "[", "]", "{", "}", '"', "'", "`"}
    count = sum(1 for tok in toks if tok in delimiters)
    return count >= 4


def is_long_range_candidate(tokenizer, input_ids: np.ndarray) -> bool:
    toks = _to_token_strings(tokenizer, input_ids)
    return len(toks) >= 64


def is_code_like_sequence(tokenizer, input_ids: np.ndarray) -> bool:
    toks = _to_token_strings(tokenizer, input_ids)
    code_markers = {
        "def",
        "class",
        "return",
        "import",
        "from",
        "if",
        "else",
        ":",
        "{",
        "}",
        ";",
    }
    count = sum(1 for tok in toks if tok in code_markers)
    return count >= 3


TASK_SLICE_REGISTRY: dict[str, Callable] = {
    "repeated_token": is_repeated_token_sequence,
    "delimiter_heavy": is_delimiter_heavy_sequence,
    "long_range": is_long_range_candidate,
    "code_like": is_code_like_sequence,
}


def filter_indices_by_task_slice(
    activation_loader,
    tokenizer,
    idx_list: list[int],
    slice_name: str,
) -> list[int]:
    if slice_name not in TASK_SLICE_REGISTRY:
        raise ValueError(
            f"Unknown slice_name={slice_name!r}. "
            f"Choose from {sorted(TASK_SLICE_REGISTRY)}"
        )

    fn = TASK_SLICE_REGISTRY[slice_name]
    kept = []

    for idx in idx_list:
        input_ids = activation_loader.get_sequence_input_ids(idx)
        if fn(tokenizer, input_ids):
            kept.append(idx)

    return kept

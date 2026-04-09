from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from routing_aware_atos.routed_types import CachedSample


@dataclass(frozen=True)
class AttributionBuildConfig:
    methods: tuple[str, ...] = ("attention_value",)
    normalize_rows: bool = True
    symmetrize_similarity: bool = False
    include_existing: bool = True


SUPPORTED_ATTRIBUTION_METHODS = {
    "attention_value",
    "residual_similarity",
    "attention_similarity_mix",
}


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return matrix / row_sums


def _safe_cosine_matrix(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity matrix with shape [target_seq, source_seq].
    Negative values are clipped to zero for routing use.
    """
    t = np.asarray(targets, dtype=np.float32)
    s = np.asarray(sources, dtype=np.float32)

    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    s_norm = np.linalg.norm(s, axis=1, keepdims=True)

    t_norm = np.where(t_norm <= 1e-8, 1.0, t_norm)
    s_norm = np.where(s_norm <= 1e-8, 1.0, s_norm)

    sims = (t @ s.T) / (t_norm * s_norm.T)
    sims = np.maximum(sims, 0.0)
    return sims.astype(np.float32)


def build_attribution_score_matrix(
    sample: CachedSample,
    source_layer: int,
    target_layer: int,
    *,
    method: str = "attention_value",
    normalize_rows: bool = True,
    symmetrize_similarity: bool = False,
) -> np.ndarray:
    """
    Build a routing score matrix [target_seq, source_seq] using residuals and
    optionally pooled attention scores already attached to the sample.
    """
    if method not in SUPPORTED_ATTRIBUTION_METHODS:
        raise ValueError(
            f"Unsupported attribution method: {method!r}. "
            f"Supported: {sorted(SUPPORTED_ATTRIBUTION_METHODS)}"
        )

    if source_layer not in sample.residuals:
        raise KeyError(f"Missing source residual layer {source_layer}")
    if target_layer not in sample.residuals:
        raise KeyError(f"Missing target residual layer {target_layer}")

    source = np.asarray(sample.residuals[source_layer], dtype=np.float32)
    target = np.asarray(sample.residuals[target_layer], dtype=np.float32)
    key = (source_layer, target_layer)

    if method == "attention_value":
        if sample.attention_scores is None or key not in sample.attention_scores:
            raise KeyError(
                f"Missing attention score matrix for layer pair {key} required by {method}"
            )

        attn = np.asarray(sample.attention_scores[key], dtype=np.float32)
        source_norm = np.linalg.norm(source, axis=1, keepdims=True).T
        matrix = attn * source_norm

    elif method == "residual_similarity":
        matrix = _safe_cosine_matrix(target, source)
        if symmetrize_similarity and source.shape == target.shape:
            rev = _safe_cosine_matrix(source, target).T
            matrix = 0.5 * (matrix + rev)

    elif method == "attention_similarity_mix":
        if sample.attention_scores is None or key not in sample.attention_scores:
            raise KeyError(
                f"Missing attention score matrix for layer pair {key} required by {method}"
            )

        attn = np.asarray(sample.attention_scores[key], dtype=np.float32)
        sims = _safe_cosine_matrix(target, source)
        matrix = 0.5 * attn + 0.5 * sims

    else:
        raise AssertionError("Unreachable")

    matrix = np.maximum(matrix, 0.0).astype(np.float32)

    if normalize_rows:
        matrix = _row_normalize(matrix)

    return matrix


def attach_attribution_scores(
    samples: Iterable[CachedSample],
    layer_pairs: list[tuple[int, int]],
    *,
    config: AttributionBuildConfig,
) -> list[CachedSample]:
    """
    Return new CachedSample objects with attribution_scores populated for each
    requested (source_layer, target_layer) pair.
    """
    new_samples: list[CachedSample] = []

    for sample in samples:
        sample.validate()

        existing = {}
        if config.include_existing and sample.attribution_scores is not None:
            existing.update(
                {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in sample.attribution_scores.items()
                }
            )

        for source_layer, target_layer in layer_pairs:
            per_method = []
            for method in config.methods:
                mat = build_attribution_score_matrix(
                    sample=sample,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    method=method,
                    normalize_rows=config.normalize_rows,
                    symmetrize_similarity=config.symmetrize_similarity,
                )
                per_method.append(mat)

            stacked = np.stack(per_method, axis=0)
            existing[(source_layer, target_layer)] = stacked.mean(axis=0).astype(
                np.float32
            )

        new_samples.append(
            CachedSample(
                tokens=sample.tokens,
                residuals=sample.residuals,
                attention_scores=sample.attention_scores,
                attribution_scores=existing,
                metadata=dict(sample.metadata or {}),
            )
        )

    return new_samples


def summarize_attribution_scores(
    samples: Iterable[CachedSample],
    layer_pair: tuple[int, int],
) -> dict[str, float]:
    matrices = []
    for sample in samples:
        if sample.attribution_scores is None:
            continue
        if layer_pair not in sample.attribution_scores:
            continue
        matrices.append(
            np.asarray(sample.attribution_scores[layer_pair], dtype=np.float32)
        )

    if not matrices:
        raise ValueError(f"No attribution matrices found for layer pair {layer_pair}")

    stack = np.stack(matrices, axis=0)
    row_entropy = -(stack * np.log(np.clip(stack, 1e-12, None))).sum(axis=-1).mean()
    top1_mass = stack.max(axis=-1).mean()

    return {
        "num_samples": float(stack.shape[0]),
        "seq_len": float(stack.shape[-1]),
        "mean_top1_mass": float(top1_mass),
        "mean_row_entropy": float(row_entropy),
        "mean_score": float(stack.mean()),
    }


__all__ = [
    "AttributionBuildConfig",
    "attach_attribution_scores",
    "build_attribution_score_matrix",
    "summarize_attribution_scores",
]

from __future__ import annotations

import numpy as np


def minmax_normalize(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    transformed = -values if invert else values
    finite = np.isfinite(transformed)
    if not finite.any():
        return np.zeros(values.size, dtype=np.float32)
    floor = float(np.min(transformed[finite]))
    ceiling = float(np.max(transformed[finite]))
    filled = np.nan_to_num(transformed, nan=floor, posinf=ceiling, neginf=floor)
    span = ceiling - floor
    if span <= 1e-12:
        return np.zeros(values.size, dtype=np.float32)
    return np.clip((filled - floor) / span, 0.0, 1.0).astype(np.float32)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    denominator = max(values.size - 1, 1)
    return (ranks / denominator).astype(np.float32)


def binned_entropy(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must have shape [features, bins]")
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)
    return entropy.astype(np.float32)


def normalized_binned_entropy(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    entropy = binned_entropy(counts)
    occupied = np.maximum((counts > 0).sum(axis=1), 2)
    return (entropy / np.log(occupied)).astype(np.float32)


def max_abs_anchor_correlation(
    centered_cross_products: np.ndarray,
    centered_sum_squares: np.ndarray,
    anchor_columns: np.ndarray,
) -> np.ndarray:
    """Return each feature's largest absolute Pearson correlation to an anchor."""

    cross_products = np.asarray(centered_cross_products, dtype=np.float64)
    sum_squares = np.asarray(centered_sum_squares, dtype=np.float64)
    anchors = np.asarray(anchor_columns, dtype=np.int64)
    if cross_products.ndim != 2 or sum_squares.ndim != 1 or anchors.ndim != 1:
        raise ValueError("correlation statistics have incompatible dimensions")
    if cross_products.shape != (anchors.size, sum_squares.size):
        raise ValueError(
            "centered_cross_products must have shape [num_anchors, num_features]"
        )
    if anchors.size == 0:
        return np.zeros(sum_squares.size, dtype=np.float32)
    if np.any(anchors < 0) or np.any(anchors >= sum_squares.size):
        raise ValueError("anchor_columns contains an out-of-range feature column")

    denominator = np.sqrt(
        np.maximum(sum_squares[anchors, None], 0.0)
        * np.maximum(sum_squares[None, :], 0.0)
    )
    correlations = np.divide(
        cross_products,
        denominator,
        out=np.zeros_like(cross_products),
        where=denominator > 1e-12,
    )
    correlations = np.clip(np.abs(correlations), 0.0, 1.0)
    correlations[np.arange(anchors.size), anchors] = 0.0
    return correlations.max(axis=0).astype(np.float32)


def composite_feature_scores(
    *,
    activation_strength: np.ndarray,
    token_entropy: np.ndarray,
    max_decoder_cosine: np.ndarray,
    coherence_weight: float = 0.45,
    uniqueness_weight: float = 0.25,
    activity_weight: float = 0.30,
) -> np.ndarray:
    total_weight = coherence_weight + uniqueness_weight + activity_weight
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"feature score weights must sum to 1.0, got {total_weight}")
    activity = percentile_rank(np.asarray(activation_strength))
    coherence = 1.0 - np.clip(np.asarray(token_entropy, dtype=np.float32), 0.0, 1.0)
    uniqueness = 1.0 - np.clip(np.asarray(max_decoder_cosine, dtype=np.float32), 0.0, 1.0)
    return (
        coherence_weight * coherence
        + uniqueness_weight * uniqueness
        + activity_weight * activity
    ).astype(np.float32)


def reference_style_feature_scores(
    *,
    token_entropy: np.ndarray,
    vocabulary_focus: np.ndarray,
    redundancy: np.ndarray,
    activation_rate: np.ndarray,
    causal_effect: np.ndarray | None = None,
    causal_candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the score weights used by the authors' released selector."""

    arrays = [
        np.asarray(token_entropy),
        np.asarray(vocabulary_focus),
        np.asarray(redundancy),
        np.asarray(activation_rate),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all feature metrics must be one-dimensional")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all feature metrics must have the same length")

    coherence = minmax_normalize(arrays[0], invert=True)
    focus = minmax_normalize(arrays[1])
    redundancy_penalty = minmax_normalize(arrays[2])
    dead_feature_penalty = minmax_normalize(arrays[3], invert=True)
    if causal_effect is None:
        return (
            0.40 * coherence
            + 0.35 * focus
            - 0.15 * redundancy_penalty
            - 0.10 * dead_feature_penalty
        ).astype(np.float32)

    causal = np.asarray(causal_effect)
    if causal.ndim != 1 or causal.size != arrays[0].size:
        raise ValueError("causal_effect must match the other feature metrics")
    if causal_candidate_mask is None:
        causal_score = minmax_normalize(np.maximum(causal, 0.0))
    else:
        candidate_mask = np.asarray(causal_candidate_mask, dtype=bool)
        if candidate_mask.ndim != 1 or candidate_mask.size != causal.size:
            raise ValueError("causal_candidate_mask must match causal_effect")
        causal_score = np.zeros(causal.size, dtype=np.float32)
        causal_score[candidate_mask] = minmax_normalize(
            np.maximum(causal[candidate_mask], 0.0)
        )
    return (
        0.35 * coherence
        + 0.25 * focus
        + 0.30 * causal_score
        - 0.07 * redundancy_penalty
        - 0.03 * dead_feature_penalty
    ).astype(np.float32)


__all__ = [
    "composite_feature_scores",
    "binned_entropy",
    "minmax_normalize",
    "max_abs_anchor_correlation",
    "normalized_binned_entropy",
    "percentile_rank",
    "reference_style_feature_scores",
]

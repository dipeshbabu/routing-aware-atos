from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def validate_sae_artifact(arrays: Mapping[str, np.ndarray]) -> None:
    if "decoder" not in arrays:
        raise ValueError("SAE artifact must contain a decoder array")
    decoder = np.asarray(arrays["decoder"])
    if decoder.ndim != 2:
        raise ValueError(f"decoder must be 2D, got {decoder.shape}")
    if "encoder" in arrays:
        encoder = np.asarray(arrays["encoder"])
        if encoder.ndim != 2:
            raise ValueError(f"encoder must be 2D, got {encoder.shape}")
        if encoder.shape != (decoder.shape[1], decoder.shape[0]):
            raise ValueError(
                f"encoder {encoder.shape} is not the transpose-compatible shape for decoder {decoder.shape}"
            )
    normalization = (
        str(np.asarray(arrays["normalize_activations"]).item())
        if "normalize_activations" in arrays
        else "none"
    )
    if normalization not in {"none", "None"}:
        raise ValueError(
            f"Unsupported SAE activation normalization {normalization!r}; export folded weights or use sae.encode()"
        )
    architecture = (
        str(np.asarray(arrays["architecture"]).item())
        if "architecture" in arrays
        else "standard"
    )
    if architecture not in {"standard", "jumprelu"}:
        raise ValueError(f"Unsupported SAE architecture {architecture!r}")
    activation_fn = (
        str(np.asarray(arrays["activation_fn"]).item())
        if "activation_fn" in arrays
        else "relu"
    )
    if activation_fn != "relu":
        raise ValueError(f"Unsupported SAE activation function {activation_fn!r}")


def sae_feature_activations(
    residuals: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    feature_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Encode Gemma Scope ReLU/JumpReLU features from an exported SAE artifact."""

    validate_sae_artifact(arrays)
    if "encoder" not in arrays:
        raise ValueError("SAE artifact must contain encoder weights to compute activations")

    x = np.asarray(residuals, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"residuals must be 2D, got {x.shape}")
    encoder = np.asarray(arrays["encoder"], dtype=np.float32)
    ids = np.arange(encoder.shape[1]) if feature_ids is None else np.asarray(feature_ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("feature_ids cannot be empty")

    centered = x
    apply_b_dec = (
        bool(np.asarray(arrays["apply_b_dec_to_input"]).item())
        if "apply_b_dec_to_input" in arrays
        else True
    )
    if "b_dec" in arrays and apply_b_dec:
        centered = centered - np.asarray(arrays["b_dec"], dtype=np.float32)
    pre = centered @ encoder[:, ids]
    if "b_enc" in arrays:
        pre = pre + np.asarray(arrays["b_enc"], dtype=np.float32)[ids]

    if "threshold" in arrays:
        threshold = np.asarray(arrays["threshold"], dtype=np.float32)
        if threshold.ndim == 0:
            selected_threshold = threshold
        else:
            selected_threshold = threshold[ids]
        return np.where(pre > selected_threshold, np.maximum(pre, 0.0), 0.0).astype(np.float32)
    return np.maximum(pre, 0.0).astype(np.float32)


def active_feature_mask(
    residuals: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    feature_ids: Sequence[int] | None = None,
) -> np.ndarray:
    return sae_feature_activations(
        residuals,
        arrays,
        feature_ids=feature_ids,
    ) > 0


__all__ = ["active_feature_mask", "sae_feature_activations", "validate_sae_artifact"]

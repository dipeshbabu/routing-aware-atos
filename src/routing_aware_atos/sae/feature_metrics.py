from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import torch

from routing_aware_atos.sae.encoding import validate_sae_artifact


@dataclass
class FeatureSpaceMetrics:
    feature_ids: list[int]
    r2: np.ndarray
    mse: np.ndarray
    corr: np.ndarray
    activation_rmse: np.ndarray
    activation_counts: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_ids": list(self.feature_ids),
            "r2": self.r2.tolist(),
            "mse": self.mse.tolist(),
            "corr": self.corr.tolist(),
            "activation_rmse": self.activation_rmse.tolist(),
            "activation_counts": self.activation_counts.astype(int).tolist(),
        }


def _selected_sae_parameters(
    sae_arrays: Mapping[str, np.ndarray],
    feature_ids: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    normalization = (
        str(np.asarray(sae_arrays["normalize_activations"]).item())
        if "normalize_activations" in sae_arrays
        else "none"
    )
    if normalization not in {"none", "None"}:
        raise ValueError(f"Unsupported SAE activation normalization {normalization!r}")
    if "encoder" not in sae_arrays:
        raise ValueError("activated_only evaluation requires encoder weights in the SAE artifact")
    ids = np.asarray(feature_ids, dtype=np.int64)
    encoder = torch.as_tensor(
        np.asarray(sae_arrays["encoder"], dtype=np.float32)[:, ids],
        device=device,
    )
    apply_b_dec = (
        bool(np.asarray(sae_arrays["apply_b_dec_to_input"]).item())
        if "apply_b_dec_to_input" in sae_arrays
        else True
    )
    b_dec = (
        torch.as_tensor(np.asarray(sae_arrays["b_dec"], dtype=np.float32), device=device)
        if "b_dec" in sae_arrays and apply_b_dec
        else None
    )
    b_enc = (
        torch.as_tensor(np.asarray(sae_arrays["b_enc"], dtype=np.float32)[ids], device=device)
        if "b_enc" in sae_arrays
        else None
    )
    threshold = None
    if "threshold" in sae_arrays:
        raw_threshold = np.asarray(sae_arrays["threshold"], dtype=np.float32)
        threshold = torch.as_tensor(
            raw_threshold if raw_threshold.ndim == 0 else raw_threshold[ids],
            device=device,
        )
    return encoder, b_dec, b_enc, threshold


def evaluate_feature_space(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    decoder_matrix: np.ndarray,
    feature_ids: Iterable[int] | None = None,
    *,
    sae_arrays: Mapping[str, np.ndarray] | None = None,
    activated_only: bool = False,
    min_activations: int = 1,
    compute_device: str = "cpu",
    batch_size: int = 2048,
    normalize_decoder: bool = True,
    min_r2: float | None = None,
) -> FeatureSpaceMetrics:
    """Evaluate decoder projections using streaming sufficient statistics."""

    Y_true = np.asarray(Y_true, dtype=np.float32)
    Y_pred = np.asarray(Y_pred, dtype=np.float32)
    decoder_matrix = np.asarray(decoder_matrix, dtype=np.float32)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Y_true and Y_pred must have identical shape, got {Y_true.shape} vs {Y_pred.shape}")
    if Y_true.ndim != 2:
        raise ValueError(f"Expected 2D Y arrays, got {Y_true.shape}")
    if decoder_matrix.ndim != 2:
        raise ValueError(f"Expected 2D decoder_matrix, got {decoder_matrix.shape}")
    if decoder_matrix.shape[1] != Y_true.shape[1]:
        raise ValueError(
            f"Decoder d_model {decoder_matrix.shape[1]} does not match Y d_model {Y_true.shape[1]}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if min_activations <= 0:
        raise ValueError("min_activations must be positive")

    ids = list(range(decoder_matrix.shape[0])) if feature_ids is None else [int(value) for value in feature_ids]
    if not ids:
        raise ValueError("feature_ids cannot be empty")
    device = torch.device(compute_device)
    decoder = torch.as_tensor(decoder_matrix[ids], dtype=torch.float32, device=device)
    if normalize_decoder:
        decoder = torch.nn.functional.normalize(decoder, dim=1)
    encoder = b_dec = b_enc = threshold = None
    if activated_only:
        if sae_arrays is None:
            raise ValueError("activated_only evaluation requires sae_arrays")
        validate_sae_artifact(sae_arrays)
        encoder, b_dec, b_enc, threshold = _selected_sae_parameters(
            sae_arrays,
            ids,
            device=device,
        )

    num_features = len(ids)
    count = torch.zeros(num_features, dtype=torch.float64, device=device)
    sum_true = torch.zeros_like(count)
    sum_pred = torch.zeros_like(count)
    sum_true_sq = torch.zeros_like(count)
    sum_pred_sq = torch.zeros_like(count)
    sum_cross = torch.zeros_like(count)
    sum_squared_error = torch.zeros_like(count)

    with torch.inference_mode():
        for start in range(0, Y_true.shape[0], batch_size):
            stop = min(start + batch_size, Y_true.shape[0])
            true = torch.as_tensor(Y_true[start:stop], dtype=torch.float32, device=device)
            pred = torch.as_tensor(Y_pred[start:stop], dtype=torch.float32, device=device)
            true_projection = true @ decoder.T
            pred_projection = pred @ decoder.T
            if activated_only:
                centered = true if b_dec is None else true - b_dec
                pre = centered @ encoder
                if b_enc is not None:
                    pre = pre + b_enc
                active = pre > (0.0 if threshold is None else threshold)
                mask = active.to(torch.float64)
            else:
                mask = torch.ones_like(true_projection, dtype=torch.float64)

            true64 = true_projection.to(torch.float64)
            pred64 = pred_projection.to(torch.float64)
            count += mask.sum(dim=0)
            sum_true += (true64 * mask).sum(dim=0)
            sum_pred += (pred64 * mask).sum(dim=0)
            sum_true_sq += (true64.square() * mask).sum(dim=0)
            sum_pred_sq += (pred64.square() * mask).sum(dim=0)
            sum_cross += (true64 * pred64 * mask).sum(dim=0)
            sum_squared_error += ((true64 - pred64).square() * mask).sum(dim=0)

    safe_count = torch.clamp(count, min=1.0)
    mean_true = sum_true / safe_count
    mean_pred = sum_pred / safe_count
    var_true = torch.clamp(sum_true_sq / safe_count - mean_true.square(), min=0.0)
    var_pred = torch.clamp(sum_pred_sq / safe_count - mean_pred.square(), min=0.0)
    covariance = sum_cross / safe_count - mean_true * mean_pred
    mse = sum_squared_error / safe_count
    r2 = torch.where(var_true > 1e-12, 1.0 - mse / var_true, torch.zeros_like(mse))
    correlation = torch.where(
        (var_true > 1e-12) & (var_pred > 1e-12),
        covariance / torch.sqrt(var_true * var_pred),
        torch.zeros_like(covariance),
    )
    valid = count >= int(min_activations)
    if min_r2 is not None:
        valid &= r2 > float(min_r2)
    valid_np = valid.detach().cpu().numpy().astype(bool)
    selected_ids = [feature_id for feature_id, keep in zip(ids, valid_np) if keep]
    if not selected_ids:
        raise ValueError(
            f"No features passed min_activations={min_activations} and min_r2={min_r2}; "
            f"maximum count was {int(count.max().item())}"
        )

    metrics = FeatureSpaceMetrics(
        feature_ids=selected_ids,
        r2=r2[valid].detach().cpu().numpy().astype(np.float32),
        mse=mse[valid].detach().cpu().numpy().astype(np.float32),
        corr=correlation[valid].detach().cpu().numpy().astype(np.float32),
        activation_rmse=torch.sqrt(mse[valid]).detach().cpu().numpy().astype(np.float32),
        activation_counts=count[valid].detach().cpu().numpy().astype(np.int64),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def summarize_feature_metrics(metrics: FeatureSpaceMetrics) -> Dict[str, float]:
    return {
        "num_features": float(len(metrics.feature_ids)),
        "mean_r2": float(np.mean(metrics.r2)),
        "median_r2": float(np.median(metrics.r2)),
        "mean_mse": float(np.mean(metrics.mse)),
        "mean_corr": float(np.mean(metrics.corr)),
        "mean_activation_rmse": float(np.mean(metrics.activation_rmse)),
        "min_activation_count": float(np.min(metrics.activation_counts)),
        "median_activation_count": float(np.median(metrics.activation_counts)),
        "top10_mean_r2": float(np.mean(np.sort(metrics.r2)[-min(10, len(metrics.r2)) :])),
    }

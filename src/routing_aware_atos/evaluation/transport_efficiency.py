from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from routing_aware_atos.models.transport_operator import TransportOperator


@dataclass
class TransportEfficiencyMetrics:
    rank: int
    residual_r2: float
    whitened_r2: float
    ceiling_r2: float
    efficiency: float
    raw_efficiency: float
    effective_dimensionality: float
    canonical_correlations: np.ndarray
    squared_canonical_correlations: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "residual_r2": float(self.residual_r2),
            "whitened_r2": float(self.whitened_r2),
            "ceiling_r2": float(self.ceiling_r2),
            "efficiency": float(self.efficiency),
            "raw_efficiency": float(self.raw_efficiency),
            "effective_dimensionality": float(self.effective_dimensionality),
            "canonical_correlations": self.canonical_correlations.tolist(),
            "squared_canonical_correlations": self.squared_canonical_correlations.tolist(),
        }


def _as_2d_float(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two rows")
    return array


def _center(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    return values - mean, mean


def _cov(values: np.ndarray) -> np.ndarray:
    return (values.T @ values) / float(values.shape[0] - 1)


def _inverse_sqrt_psd(cov: np.ndarray, eps: float) -> np.ndarray:
    eps = float(eps)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    threshold = eps * max(max_eval, 1.0)
    inv = np.zeros_like(evals)
    keep = evals > threshold
    inv[keep] = 1.0 / np.sqrt(evals[keep])
    return (evecs * inv) @ evecs.T


def canonical_correlations(X: np.ndarray, Y: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    X = _as_2d_float("X", X)
    Y = _as_2d_float("Y", Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same rows, got {X.shape[0]} and {Y.shape[0]}")

    Xc, _ = _center(X)
    Yc, _ = _center(Y)
    cov_xx = _cov(Xc)
    cov_yy = _cov(Yc)
    cov_xy = (Xc.T @ Yc) / float(Xc.shape[0] - 1)

    wx = _inverse_sqrt_psd(cov_xx, eps)
    wy = _inverse_sqrt_psd(cov_yy, eps)
    whitened_cross_cov = wx @ cov_xy @ wy
    singular_values = np.linalg.svd(whitened_cross_cov, compute_uv=False)
    return np.clip(singular_values, 0.0, 1.0)


def transport_r2_ceiling(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    rank: int,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    Y = _as_2d_float("Y", Y)
    rhos = canonical_correlations(X, Y, eps=eps)
    squared = rhos**2
    capped_rank = min(rank, squared.size)
    ceiling = float(np.sum(squared[:capped_rank]) / Y.shape[1])
    return ceiling, rhos


def effective_transport_dimensionality(squared_correlations: np.ndarray) -> float:
    squared = np.asarray(squared_correlations, dtype=np.float64)
    numerator = float(np.sum(squared) ** 2)
    denominator = float(np.sum(squared**2))
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def whitened_target_r2(Y_true: np.ndarray, Y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    Y_true = _as_2d_float("Y_true", Y_true)
    Y_pred = _as_2d_float("Y_pred", Y_pred)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Y_true and Y_pred must match, got {Y_true.shape} and {Y_pred.shape}")

    Yc, mean = _center(Y_true)
    pred_c = Y_pred - mean
    whitening = _inverse_sqrt_psd(_cov(Yc), eps)
    Yw = Yc @ whitening
    pred_w = pred_c @ whitening

    total = float(np.sum(Yw**2))
    if total <= 1e-12:
        return 0.0
    error = float(np.sum((Yw - pred_w) ** 2))
    return 1.0 - error / total


def residual_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    Y_true = _as_2d_float("Y_true", Y_true)
    Y_pred = _as_2d_float("Y_pred", Y_pred)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"Y_true and Y_pred must match, got {Y_true.shape} and {Y_pred.shape}")
    total = float(np.sum((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2))
    if total <= 1e-12:
        return 0.0
    error = float(np.sum((Y_true - Y_pred) ** 2))
    return 1.0 - error / total


def compute_transport_efficiency(
    X: np.ndarray,
    Y: np.ndarray,
    Y_pred: np.ndarray,
    *,
    rank: int,
    eps: float = 1e-8,
) -> TransportEfficiencyMetrics:
    rank = int(rank)
    ceiling, rhos = transport_r2_ceiling(X, Y, rank=rank, eps=eps)
    squared = rhos**2
    whitened_r2 = whitened_target_r2(Y, Y_pred, eps=eps)
    raw_efficiency = 0.0 if ceiling <= 1e-12 else float(whitened_r2 / ceiling)
    efficiency = float(np.clip(raw_efficiency, 0.0, 1.0))

    return TransportEfficiencyMetrics(
        rank=int(rank),
        residual_r2=residual_r2(Y, Y_pred),
        whitened_r2=float(whitened_r2),
        ceiling_r2=float(ceiling),
        efficiency=efficiency,
        raw_efficiency=raw_efficiency,
        effective_dimensionality=effective_transport_dimensionality(squared),
        canonical_correlations=rhos.astype(np.float64),
        squared_canonical_correlations=squared.astype(np.float64),
    )


def evaluate_operator_transport_efficiency(
    operator: TransportOperator,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    rank: int | None = None,
    eps: float = 1e-8,
) -> TransportEfficiencyMetrics:
    if rank is None:
        rank = operator.config.rank
    if rank is None:
        rank = min(np.asarray(X).shape[1], np.asarray(Y).shape[1])
    Y_pred = operator.predict(X)
    return compute_transport_efficiency(X, Y, Y_pred, rank=int(rank), eps=eps)

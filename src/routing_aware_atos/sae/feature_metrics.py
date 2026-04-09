from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class FeatureSpaceMetrics:
    feature_ids: List[int]
    r2: np.ndarray
    mse: np.ndarray
    corr: np.ndarray
    activation_rmse: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            'feature_ids': list(self.feature_ids),
            'r2': self.r2.tolist(),
            'mse': self.mse.tolist(),
            'corr': self.corr.tolist(),
            'activation_rmse': self.activation_rmse.tolist(),
        }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.var(y_true))
    if var <= 1e-12:
        return 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    return float(1.0 - mse / var)


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if std_true <= 1e-12 or std_pred <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_feature_space(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    decoder_matrix: np.ndarray,
    feature_ids: Iterable[int] | None = None,
) -> FeatureSpaceMetrics:
    Y_true = np.asarray(Y_true, dtype=np.float32)
    Y_pred = np.asarray(Y_pred, dtype=np.float32)
    decoder_matrix = np.asarray(decoder_matrix, dtype=np.float32)

    if Y_true.shape != Y_pred.shape:
        raise ValueError(f'Y_true and Y_pred must have identical shape, got {Y_true.shape} vs {Y_pred.shape}')
    if Y_true.ndim != 2:
        raise ValueError(f'Expected 2D Y arrays, got {Y_true.shape}')
    if decoder_matrix.ndim != 2:
        raise ValueError(f'Expected 2D decoder_matrix, got {decoder_matrix.shape}')
    if decoder_matrix.shape[1] != Y_true.shape[1]:
        raise ValueError(
            f'Decoder d_model {decoder_matrix.shape[1]} does not match Y d_model {Y_true.shape[1]}'
        )

    if feature_ids is None:
        feature_ids = list(range(decoder_matrix.shape[0]))
    else:
        feature_ids = list(feature_ids)
    if not feature_ids:
        raise ValueError('feature_ids cannot be empty')

    selected = decoder_matrix[feature_ids]  # [F, d_model]
    A_true = Y_true @ selected.T
    A_pred = Y_pred @ selected.T

    r2 = np.asarray([_safe_r2(A_true[:, i], A_pred[:, i]) for i in range(A_true.shape[1])], dtype=np.float32)
    mse = np.mean((A_true - A_pred) ** 2, axis=0).astype(np.float32)
    corr = np.asarray([_safe_corr(A_true[:, i], A_pred[:, i]) for i in range(A_true.shape[1])], dtype=np.float32)
    activation_rmse = np.sqrt(mse).astype(np.float32)

    return FeatureSpaceMetrics(
        feature_ids=feature_ids,
        r2=r2,
        mse=mse,
        corr=corr,
        activation_rmse=activation_rmse,
    )


def summarize_feature_metrics(metrics: FeatureSpaceMetrics) -> Dict[str, float]:
    return {
        'num_features': float(len(metrics.feature_ids)),
        'mean_r2': float(np.mean(metrics.r2)),
        'median_r2': float(np.median(metrics.r2)),
        'mean_mse': float(np.mean(metrics.mse)),
        'mean_corr': float(np.mean(metrics.corr)),
        'mean_activation_rmse': float(np.mean(metrics.activation_rmse)),
        'top10_mean_r2': float(np.mean(np.sort(metrics.r2)[-min(10, len(metrics.r2)) :])),
    }

from __future__ import annotations

import numpy as np

from routing_aware_atos.sae.feature_metrics import evaluate_feature_space, summarize_feature_metrics


def test_feature_metrics_shapes_and_summary():
    rng = np.random.default_rng(0)
    Y_true = rng.normal(size=(20, 5)).astype(np.float32)
    Y_pred = Y_true + 0.05 * rng.normal(size=(20, 5)).astype(np.float32)
    decoder = rng.normal(size=(4, 5)).astype(np.float32)

    metrics = evaluate_feature_space(Y_true, Y_pred, decoder)
    assert len(metrics.feature_ids) == 4
    assert metrics.r2.shape == (4,)
    summary = summarize_feature_metrics(metrics)
    assert summary['mean_r2'] <= 1.0
    assert summary['mean_corr'] <= 1.0


def test_feature_metrics_normalize_decoder_directions_by_default():
    rng = np.random.default_rng(3)
    y_true = rng.normal(size=(30, 3)).astype(np.float32)
    y_pred = y_true + 0.1 * rng.normal(size=(30, 3)).astype(np.float32)
    decoder = rng.normal(size=(2, 3)).astype(np.float32)

    original = evaluate_feature_space(y_true, y_pred, decoder)
    rescaled = evaluate_feature_space(y_true, y_pred, decoder * 17.0)

    np.testing.assert_allclose(original.mse, rescaled.mse, rtol=1e-5, atol=1e-6)


def test_feature_metrics_apply_paper_r2_floor():
    y_true = np.asarray([[1.0, 1.0], [2.0, -1.0], [3.0, 2.0]], dtype=np.float32)
    y_pred = np.column_stack([y_true[:, 0], -y_true[:, 1]]).astype(np.float32)
    metrics = evaluate_feature_space(
        y_true,
        y_pred,
        np.eye(2, dtype=np.float32),
        min_r2=-1.0,
    )
    assert metrics.feature_ids == [0]

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

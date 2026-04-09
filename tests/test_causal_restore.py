from __future__ import annotations

import numpy as np

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.evaluation.causal_restore import (
    compare_causal_policy_runs,
    compute_feature_restoration,
    compute_logit_restoration,
    compute_residual_restoration,
    evaluate_causal_restoration_from_cached_samples,
)
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.utils.io import save_cached_samples, save_npz


def test_compute_residual_restoration():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = 0.5 * y_true
    metrics = compute_residual_restoration(y_true, y_pred)
    assert metrics["ablated_mse"] > metrics["restored_mse"]
    assert 0.0 < metrics["mse_restoration"] <= 1.0


def test_compute_feature_and_logit_restoration():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y_pred = 0.8 * y_true
    decoder = np.eye(2, dtype=np.float32)
    readout = np.array([[1.0, 0.5], [0.25, 1.0]], dtype=np.float32)

    feat = compute_feature_restoration(y_true, y_pred, decoder)
    logits = compute_logit_restoration(y_true, y_pred, readout)
    assert feat["feature_mse_restoration"] > 0
    assert logits["logit_mse_restoration"] > 0


def test_compare_causal_policy_runs(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4)).astype(np.float32)
    W = rng.normal(size=(4, 4)).astype(np.float32)
    Y = X @ W

    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit(X, Y)
    operator_path = tmp_path / "op.npz"
    op.save(operator_path)

    pairs_path = tmp_path / "pairs.npz"
    decoder_path = tmp_path / "decoder.npz"
    readout_path = tmp_path / "readout.npz"
    save_npz(pairs_path, X=X, Y=Y)
    save_npz(decoder_path, decoder=np.eye(4, dtype=np.float32))
    save_npz(readout_path, readout=np.eye(4, dtype=np.float32))

    payload = compare_causal_policy_runs(
        [
            {
                "policy_name": "same_token",
                "operator_path": operator_path,
                "pairs_path": pairs_path,
                "decoder_path": decoder_path,
                "readout_path": readout_path,
                "rank": None,
            }
        ]
    )
    assert len(payload["runs"]) == 1
    row = payload["summary_rows"][0]
    assert row["residual_mse_restoration"] > 0.99
    assert row["feature_mse_restoration"] > 0.99


def test_evaluate_causal_restoration_from_cached_samples(tmp_path):
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=4)
    cache_path = tmp_path / "samples.json"
    decoder_path = tmp_path / "decoder.npz"
    readout_path = tmp_path / "readout.npz"
    model_path = tmp_path / "operator.npz"
    save_cached_samples(cache_path, samples)
    save_npz(decoder_path, decoder=np.eye(4, dtype=np.float32))
    save_npz(readout_path, readout=np.eye(4, dtype=np.float32))

    X = np.concatenate([sample.residuals[10] for sample in samples], axis=0)
    Y = np.concatenate([sample.residuals[12] for sample in samples], axis=0)
    op = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6)).fit(X, Y)
    op.save(model_path)

    payload = evaluate_causal_restoration_from_cached_samples(
        model_path,
        decoder_path,
        cache_path=cache_path,
        source_layer=10,
        target_layer=12,
        routing_policy="same_token",
        readout_path=readout_path,
    )
    assert payload["routing_enabled"] is True
    assert payload["routing_policy"] == "same_token"
    assert payload["residual_restoration"]["mse_restoration"] > 0.0

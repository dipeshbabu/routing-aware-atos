from __future__ import annotations

import numpy as np

from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.evaluation.feature_eval import (
    evaluate_operator_from_cached_samples,
    evaluate_operator_in_feature_space,
)
from routing_aware_atos.models.transport_operator import TransportOperator, TransportOperatorConfig
from routing_aware_atos.utils.io import save_cached_samples, save_npz


def test_evaluate_operator_in_feature_space(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    W = rng.normal(size=(4, 4)).astype(np.float32)
    Y = X @ W
    decoder = rng.normal(size=(3, 4)).astype(np.float32)

    pair_path = tmp_path / 'pairs.npz'
    decoder_path = tmp_path / 'decoder.npz'
    model_path = tmp_path / 'operator.npz'
    save_npz(pair_path, X=X, Y=Y)
    save_npz(decoder_path, decoder=decoder)

    model = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6, rank=None)).fit(X, Y)
    model.save(model_path)

    payload = evaluate_operator_in_feature_space(model_path, pair_path, decoder_path)
    assert 'feature_summary' in payload
    assert payload['feature_summary']['mean_r2'] > 0.99


def test_evaluate_operator_from_cached_samples(tmp_path):
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=4)
    cache_path = tmp_path / "samples.json"
    decoder_path = tmp_path / "decoder.npz"
    model_path = tmp_path / "operator.npz"
    save_cached_samples(cache_path, samples)

    X = np.concatenate([sample.residuals[10] for sample in samples], axis=0)
    Y = np.concatenate([sample.residuals[12] for sample in samples], axis=0)
    decoder = np.eye(4, dtype=np.float32)
    save_npz(decoder_path, decoder=decoder)

    model = TransportOperator(TransportOperatorConfig(ridge_lambda=1e-6, rank=None)).fit(X, Y)
    model.save(model_path)

    payload = evaluate_operator_from_cached_samples(
        model_path,
        decoder_path,
        cache_path=cache_path,
        source_layer=10,
        target_layer=12,
        routing_policy="same_token",
    )
    assert payload["routing_enabled"] is True
    assert payload["routing_policy"] == "same_token"
    assert "route_summary" in payload

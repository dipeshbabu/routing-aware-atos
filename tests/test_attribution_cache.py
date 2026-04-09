from __future__ import annotations

import numpy as np

from routing_aware_atos.data.attribution_cache import (
    AttributionBuildConfig,
    attach_attribution_scores,
    build_attribution_score_matrix,
    summarize_attribution_scores,
)
from routing_aware_atos.data.mock_cache import make_mock_samples


def test_build_attribution_score_matrix_attention_value():
    sample = make_mock_samples(num_samples=1, seq_len=5, d_model=4)[0]
    mat = build_attribution_score_matrix(sample, 10, 12, method="attention_value")
    assert mat.shape == (5, 5)
    np.testing.assert_allclose(mat.sum(axis=1), np.ones(5), atol=1e-5)


def test_attach_attribution_scores_adds_layer_pair():
    samples = make_mock_samples(num_samples=2, seq_len=5, d_model=4)
    cfg = AttributionBuildConfig(methods=["attention_value", "residual_similarity"])
    updated = attach_attribution_scores(samples, [(10, 12)], config=cfg)
    assert len(updated) == 2
    assert updated[0].attribution_scores is not None
    assert (10, 12) in updated[0].attribution_scores
    assert updated[0].attribution_scores[(10, 12)].shape == (5, 5)


def test_summarize_attribution_scores():
    samples = make_mock_samples(num_samples=2, seq_len=4, d_model=3)
    cfg = AttributionBuildConfig(methods=["attention_similarity_mix"])
    updated = attach_attribution_scores(samples, [(10, 12)], config=cfg)
    summary = summarize_attribution_scores(updated, (10, 12))
    assert summary["num_samples"] == 2
    assert 0.0 <= summary["mean_top1_mass"] <= 1.0

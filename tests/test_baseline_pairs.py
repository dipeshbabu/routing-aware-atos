import numpy as np

from routing_aware_atos.data.baseline_pairs import SameTokenBaselineBuilder
from routing_aware_atos.data.mock_cache import make_mock_samples


def test_baseline_builder_shapes():
    samples = make_mock_samples(num_samples=2, seq_len=6, d_model=4)
    builder = SameTokenBaselineBuilder(samples=samples, source_layer=10, target_layer=12)
    X, Y, meta = builder.build_pairs()
    assert X.shape == (12, 4)
    assert Y.shape == (12, 4)
    assert len(meta) == 12


def test_baseline_builder_uses_same_position_pairs():
    samples = make_mock_samples(num_samples=1, seq_len=4, d_model=2)
    builder = SameTokenBaselineBuilder(samples=samples, source_layer=10, target_layer=12)
    X, _, _ = builder.build_pairs()
    assert np.allclose(X, samples[0].residuals[10])

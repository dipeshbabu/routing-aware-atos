from pathlib import Path

import numpy as np

from routing_aware_atos.activation_loader import ActivationLoader
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.utils.io import save_cached_samples


def test_activation_loader_loads_sample_subsets(tmp_path: Path):
    samples = make_mock_samples(num_samples=2, seq_len=4, d_model=3)
    cache_path = tmp_path / "samples.json"
    save_cached_samples(cache_path, samples)

    loader = ActivationLoader(samples_path=cache_path)
    sample = loader.load_cached_sample(0, layers=[10], attention_layer_pairs=[(10, 12)])

    assert set(sample.residuals) == {10}
    assert sample.attention_scores is not None
    assert set(sample.attention_scores) == {(10, 12)}
    assert sample.attribution_scores is None


def test_activation_loader_iterates_in_memory_samples():
    samples = make_mock_samples(num_samples=3, seq_len=5, d_model=2)
    loader = ActivationLoader(samples=samples)

    loaded = list(loader.iter_cached_samples(layers=[10, 12]))
    assert len(loaded) == 3
    assert all(set(sample.residuals) == {10, 12} for sample in loaded)


def test_activation_loader_returns_attention_matrix():
    samples = make_mock_samples(num_samples=1, seq_len=4, d_model=2)
    loader = ActivationLoader(samples=samples)

    matrix = loader.get_attention_matrix(0, 10, 12)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.sum(axis=1), 1.0)

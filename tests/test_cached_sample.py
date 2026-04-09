import pytest
import numpy as np

from routing_aware_atos.utils.types import CachedSample


def test_cached_sample_validates_shapes():
    sample = CachedSample(
        tokens=["a", "b"],
        residuals={0: np.zeros((2, 4), dtype=np.float32), 1: np.zeros((2, 4), dtype=np.float32)},
    )
    sample.validate()


def test_cached_sample_rejects_mismatched_token_length():
    sample = CachedSample(
        tokens=["a"],
        residuals={0: np.zeros((2, 4), dtype=np.float32)},
    )
    with pytest.raises(ValueError):
        sample.validate()

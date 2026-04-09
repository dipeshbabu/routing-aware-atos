from __future__ import annotations

from typing import List

import numpy as np

from routing_aware_atos.utils.types import CachedSample


def make_mock_samples(num_samples: int = 2, seq_len: int = 6, d_model: int = 4) -> List[CachedSample]:
    rng = np.random.default_rng(0)
    samples = []
    for sample_idx in range(num_samples):
        tokens = [f"tok_{sample_idx}_{i}" for i in range(seq_len)]
        residual_l = rng.normal(size=(seq_len, d_model)).astype(np.float32)
        residual_t = rng.normal(size=(seq_len, d_model)).astype(np.float32)

        attn = rng.uniform(size=(seq_len, seq_len)).astype(np.float32)
        attn = attn / attn.sum(axis=1, keepdims=True)

        attr = rng.uniform(size=(seq_len, seq_len)).astype(np.float32)
        attr = attr / attr.sum(axis=1, keepdims=True)

        samples.append(
            CachedSample(
                tokens=tokens,
                residuals={10: residual_l, 12: residual_t},
                attention_scores={(10, 12): attn},
                attribution_scores={(10, 12): attr},
                metadata={"sample_idx": sample_idx},
            )
        )
    return samples

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from routing_aware_atos.utils.types import CachedSample


class SameTokenBaselineBuilder:
    """Minimal baseline pair builder aligned with the original same-token ATO assumption."""

    def __init__(self, samples: Iterable[CachedSample], source_layer: int, target_layer: int, include_positions: Optional[List[int]] = None):
        self.samples = list(samples)
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.include_positions = include_positions

    def build_pairs(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        X_rows = []
        Y_rows = []
        metadata = []

        for sample_idx, sample in enumerate(self.samples):
            sample.validate()
            H_src = sample.residuals[self.source_layer]
            H_tgt = sample.residuals[self.target_layer]
            positions = self.include_positions if self.include_positions is not None else list(range(sample.seq_len))

            for pos in positions:
                X_rows.append(H_src[pos].astype(np.float32, copy=False))
                Y_rows.append(H_tgt[pos].astype(np.float32, copy=False))
                metadata.append(
                    {
                        "sample_idx": sample_idx,
                        "target_pos": pos,
                        "source_pos": pos,
                        "token": sample.tokens[pos],
                    }
                )

        return np.stack(X_rows), np.stack(Y_rows), metadata


def build_same_token_pairs(
    samples: Iterable[CachedSample],
    source_layer: int,
    target_layer: int,
    include_positions: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    builder = SameTokenBaselineBuilder(
        samples=samples,
        source_layer=source_layer,
        target_layer=target_layer,
        include_positions=include_positions,
    )
    return builder.build_pairs()

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from routing_aware_atos.routing.base import RoutingPolicy
from routing_aware_atos.utils.types import CachedSample, RouteSelection


@dataclass
class RoutedPairs:
    X: np.ndarray
    Y: np.ndarray
    routes: List[dict]


class RoutedActivationDataset:
    def __init__(
        self,
        samples: Iterable[CachedSample],
        source_layer: int,
        target_layer: int,
        routing_policy: RoutingPolicy,
        include_positions: Optional[List[int]] = None,
    ):
        self.samples = list(samples)
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.routing_policy = routing_policy
        self.include_positions = include_positions

    def build_pairs(self) -> RoutedPairs:
        X_rows: List[np.ndarray] = []
        Y_rows: List[np.ndarray] = []
        routes: List[dict] = []

        for sample_idx, sample in enumerate(self.samples):
            sample.validate()
            if self.source_layer not in sample.residuals:
                raise KeyError(f"Source layer {self.source_layer} missing from sample {sample_idx}")
            if self.target_layer not in sample.residuals:
                raise KeyError(f"Target layer {self.target_layer} missing from sample {sample_idx}")

            H_src = sample.residuals[self.source_layer]
            H_tgt = sample.residuals[self.target_layer]
            positions = self.include_positions if self.include_positions is not None else list(range(sample.seq_len))

            for target_pos in positions:
                selection = self.routing_policy.select_sources(
                    sample=sample,
                    target_pos=target_pos,
                    source_layer=self.source_layer,
                    target_layer=self.target_layer,
                )
                x_i = self._collapse_sources(H_src, selection)
                y_i = H_tgt[target_pos]

                X_rows.append(x_i.astype(np.float32, copy=False))
                Y_rows.append(y_i.astype(np.float32, copy=False))
                routes.append(
                    {
                        "sample_idx": sample_idx,
                        "target_pos": target_pos,
                        "source_ids": selection.source_ids,
                        "source_weights": selection.source_weights,
                        "score_type": selection.score_type,
                        "tokens": [sample.tokens[j] for j in selection.source_ids],
                        "target_token": sample.tokens[target_pos],
                    }
                )

        return RoutedPairs(X=np.stack(X_rows), Y=np.stack(Y_rows), routes=routes)

    @staticmethod
    def _collapse_sources(H_src: np.ndarray, selection: RouteSelection) -> np.ndarray:
        out = np.zeros(H_src.shape[1], dtype=np.float64)
        for src_idx, weight in zip(selection.source_ids, selection.source_weights):
            out += float(weight) * H_src[src_idx]
        return out.astype(np.float32)

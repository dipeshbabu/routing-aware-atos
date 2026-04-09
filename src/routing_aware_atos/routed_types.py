from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

LayerPair = Tuple[int, int]


@dataclass(frozen=True)
class RouteSelection:
    source_ids: List[int]
    source_weights: List[float]
    score_type: str


@dataclass
class CachedSample:
    """
    Lightweight container for one cached sequence worth of routing-ready data.

    Attributes:
        tokens:
            Token ids or token strings for the sequence.
        residuals:
            Mapping from layer index -> residual stream activations of shape [seq_len, d_model].
        attention_scores:
            Optional mapping from (source_layer, target_layer) -> score matrix [target_seq, source_seq].
            This can be raw attention, pooled attention, or any routing score matrix.
        attribution_scores:
            Optional mapping from (source_layer, target_layer) -> score matrix [target_seq, source_seq].
        metadata:
            Optional free-form metadata for debugging / analysis.
    """

    tokens: Sequence[str] | Sequence[int]
    residuals: Mapping[int, np.ndarray]
    attention_scores: Optional[Mapping[LayerPair, np.ndarray]] = None
    attribution_scores: Optional[Mapping[LayerPair, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if not self.residuals:
            raise ValueError("CachedSample.residuals cannot be empty")

        seq_lens = {arr.shape[0] for arr in self.residuals.values()}
        if len(seq_lens) != 1:
            raise ValueError(
                f"All residual arrays must share the same seq_len, got {seq_lens}"
            )

        d_models = {arr.shape[1] for arr in self.residuals.values()}
        if len(d_models) != 1:
            raise ValueError(
                f"All residual arrays must share the same d_model, got {d_models}"
            )

        seq_len = next(iter(seq_lens))
        if len(self.tokens) != seq_len:
            raise ValueError(
                f"tokens length {len(self.tokens)} does not match residual seq_len {seq_len}"
            )

        for layer_idx, arr in self.residuals.items():
            if arr.ndim != 2:
                raise ValueError(
                    f"Residual array for layer {layer_idx} must be 2D, got shape {arr.shape}"
                )

        for name, score_map in (
            ("attention_scores", self.attention_scores),
            ("attribution_scores", self.attribution_scores),
        ):
            if score_map is None:
                continue
            for key, matrix in score_map.items():
                if matrix.ndim != 2:
                    raise ValueError(
                        f"{name}[{key}] must be 2D [target_seq, source_seq], got {matrix.shape}"
                    )
                if matrix.shape[0] != seq_len or matrix.shape[1] != seq_len:
                    raise ValueError(
                        f"{name}[{key}] must have shape [{seq_len}, {seq_len}], got {matrix.shape}"
                    )

    @property
    def seq_len(self) -> int:
        self.validate()
        return next(iter(self.residuals.values())).shape[0]

    @property
    def d_model(self) -> int:
        self.validate()
        return next(iter(self.residuals.values())).shape[1]


@dataclass
class RoutedPairs:
    """
    Training / eval pairs for routed transport.

    Attributes:
        X:
            Routed upstream vectors [N, d_model]
        Y:
            Downstream target vectors [N, d_model]
        routes:
            Per-example route metadata
    """

    X: np.ndarray
    Y: np.ndarray
    routes: list[dict[str, Any]]

    def validate(self) -> None:
        if self.X.ndim != 2 or self.Y.ndim != 2:
            raise ValueError(
                f"RoutedPairs expects 2D arrays, got X{self.X.shape}, Y{self.Y.shape}"
            )
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of rows, got {self.X.shape[0]} vs {self.Y.shape[0]}"
            )
        if self.X.shape[1] != self.Y.shape[1]:
            raise ValueError(
                f"X and Y must have same feature dimension, got {self.X.shape[1]} vs {self.Y.shape[1]}"
            )
        if len(self.routes) != self.X.shape[0]:
            raise ValueError(
                f"routes length {len(self.routes)} must match number of rows {self.X.shape[0]}"
            )

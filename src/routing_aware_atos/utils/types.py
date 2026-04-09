from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LayerPair = Tuple[int, int]


@dataclass(frozen=True)
class RouteSelection:
    source_ids: List[int]
    source_weights: List[float]
    score_type: str


@dataclass
class CachedSample:
    tokens: Sequence[str] | Sequence[int]
    residuals: Mapping[int, np.ndarray]
    attention_scores: Optional[Mapping[LayerPair, np.ndarray]] = None
    attribution_scores: Optional[Mapping[LayerPair, np.ndarray]] = None
    metadata: Optional[Dict[str, object]] = None

    def validate(self) -> None:
        if not self.residuals:
            raise ValueError("CachedSample.residuals cannot be empty")

        seq_lens = {arr.shape[0] for arr in self.residuals.values()}
        if len(seq_lens) != 1:
            raise ValueError(f"All residual arrays must share the same seq_len, got {seq_lens}")

        d_models = {arr.shape[1] for arr in self.residuals.values()}
        if len(d_models) != 1:
            raise ValueError(f"All residual arrays must share d_model, got {d_models}")

        seq_len = next(iter(seq_lens))
        if len(self.tokens) != seq_len:
            raise ValueError(f"tokens length {len(self.tokens)} does not match residual seq_len {seq_len}")

    @property
    def seq_len(self) -> int:
        self.validate()
        return next(iter(self.residuals.values())).shape[0]

    @property
    def d_model(self) -> int:
        self.validate()
        return next(iter(self.residuals.values())).shape[1]

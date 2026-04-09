
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from routing_aware_atos.models.rank_truncation import truncate_matrix_rank


@dataclass
class TransportOperatorConfig:
    ridge_lambda: float = 1e-2
    rank: int | None = None
    regression: str = "ridge"
    name: str = "transport_operator"

    def validate(self) -> None:
        if self.regression != "ridge":
            raise ValueError(f"Only ridge regression is supported in Phase 3, got {self.regression!r}")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("rank must be positive when provided")


@dataclass
class TransportOperator:
    config: TransportOperatorConfig = field(default_factory=TransportOperatorConfig)
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None
    x_mean: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    train_metrics: Dict[str, float] = field(default_factory=dict)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TransportOperator":
        self.config.validate()
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got X{X.shape}, Y{Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same number of rows, got {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset")

        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        Xc = X - x_mean
        Yc = Y - y_mean

        d_in = Xc.shape[1]
        lhs = Xc.T @ Xc + self.config.ridge_lambda * np.eye(d_in, dtype=np.float64)
        rhs = Xc.T @ Yc
        weight = np.linalg.solve(lhs, rhs)
        weight = truncate_matrix_rank(weight, self.config.rank)
        bias = y_mean - x_mean @ weight

        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.x_mean = x_mean.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.train_metrics = self.evaluate(X, Y)
        self.train_metrics["effective_rank"] = float(np.linalg.matrix_rank(self.weight))
        self.train_metrics["requested_rank"] = -1.0 if self.config.rank is None else float(self.config.rank)
        return self

    def fit_xy(self, X: np.ndarray, Y: np.ndarray) -> "TransportOperator":
        return self.fit(X, Y)

    def fit_X_y(self, X: np.ndarray, Y: np.ndarray) -> "TransportOperator":
        return self.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weight is None or self.bias is None:
            raise ValueError("TransportOperator must be fit before predict()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.shape}")
        return X @ self.weight + self.bias

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X) if self.weight is not None else None
        if preds is None:
            raise ValueError("TransportOperator must be fit before evaluate()")
        Y = np.asarray(Y, dtype=np.float32)
        mse = float(np.mean((preds - Y) ** 2))
        mae = float(np.mean(np.abs(preds - Y)))
        var = float(np.var(Y))
        r2 = 0.0 if var <= 0 else float(1.0 - mse / var)
        return {"mse": mse, "mae": mae, "r2": r2}

    def evaluate_xy(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        return self.evaluate(X, Y)

    def save(self, path: str | Path) -> None:
        if self.weight is None or self.bias is None or self.x_mean is None or self.y_mean is None:
            raise ValueError("Cannot save an unfitted operator")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weight=self.weight,
            bias=self.bias,
            x_mean=self.x_mean,
            y_mean=self.y_mean,
            ridge_lambda=np.asarray(self.config.ridge_lambda, dtype=np.float32),
            rank=np.asarray(-1 if self.config.rank is None else self.config.rank, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path, *, name: str = "transport_operator") -> "TransportOperator":
        data = np.load(path)
        rank = int(data["rank"])
        config = TransportOperatorConfig(
            ridge_lambda=float(data["ridge_lambda"]),
            rank=None if rank < 0 else rank,
            name=name,
        )
        model = cls(config=config)
        model.weight = data["weight"].astype(np.float32)
        model.bias = data["bias"].astype(np.float32)
        model.x_mean = data["x_mean"].astype(np.float32)
        model.y_mean = data["y_mean"].astype(np.float32)
        return model

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "ridge_lambda": self.config.ridge_lambda,
            "rank": self.config.rank,
            "train_metrics": self.train_metrics,
        }


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from routing_aware_atos.models.rank_truncation import truncate_matrix_rank
from routing_aware_atos.utils.io import load_npz, save_npz


@dataclass
class TransportOperatorConfig:
    ridge_lambda: float = 1e-2
    rank: int | None = None
    regression: str = "ridge"
    name: str = "transport_operator"
    compute_backend: str = "numpy"
    device: str = "cpu"

    def validate(self) -> None:
        if self.regression != "ridge":
            raise ValueError(f"Only ridge regression is supported in Phase 3, got {self.regression!r}")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("rank must be positive when provided")
        if self.compute_backend not in {"numpy", "torch"}:
            raise ValueError("compute_backend must be 'numpy' or 'torch'")


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
        if self.config.compute_backend == "torch":
            return self._fit_torch(X, Y)

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
        self.train_metrics["effective_rank"] = float(
            min(self.weight.shape)
            if self.config.rank is None
            else min(int(self.config.rank), *self.weight.shape)
        )
        self.train_metrics["requested_rank"] = -1.0 if self.config.rank is None else float(self.config.rank)
        return self

    def _fit_torch(self, X: np.ndarray, Y: np.ndarray) -> "TransportOperator":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a core dependency
            raise ImportError("The torch compute backend requires PyTorch") from exc

        X_np = np.asarray(X, dtype=np.float32)
        Y_np = np.asarray(Y, dtype=np.float32)
        if X_np.ndim != 2 or Y_np.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got X{X_np.shape}, Y{Y_np.shape}")
        if X_np.shape[0] != Y_np.shape[0]:
            raise ValueError(f"X and Y must have same number of rows, got {X_np.shape[0]} vs {Y_np.shape[0]}")
        if X_np.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset")

        device = torch.device(self.config.device)
        X_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        Y_t = torch.as_tensor(Y_np, dtype=torch.float32, device=device)
        x_mean_t = X_t.mean(dim=0)
        y_mean_t = Y_t.mean(dim=0)
        Xc = X_t - x_mean_t
        Yc = Y_t - y_mean_t
        lhs = Xc.T @ Xc
        lhs.diagonal().add_(float(self.config.ridge_lambda))
        rhs = Xc.T @ Yc
        weight_t = torch.linalg.solve(lhs, rhs)

        if self.config.rank is not None:
            max_rank = min(weight_t.shape)
            rank = min(int(self.config.rank), max_rank)
            U, singular_values, Vh = torch.linalg.svd(weight_t, full_matrices=False)
            weight_t = (U[:, :rank] * singular_values[:rank]) @ Vh[:rank, :]

        bias_t = y_mean_t - x_mean_t @ weight_t
        self.weight = weight_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self.bias = bias_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self.x_mean = x_mean_t.detach().cpu().numpy().astype(np.float32, copy=False)
        self.y_mean = y_mean_t.detach().cpu().numpy().astype(np.float32, copy=False)
        del X_t, Y_t, Xc, Yc, lhs, rhs, weight_t, bias_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self.train_metrics = self.evaluate(X_np, Y_np)
        self.train_metrics["effective_rank"] = float(
            min(self.weight.shape)
            if self.config.rank is None
            else min(int(self.config.rank), *self.weight.shape)
        )
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
        residual_sum_squares = np.sum((preds - Y) ** 2, axis=0)
        total_sum_squares = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
        per_output_r2 = np.where(
            total_sum_squares > 1e-12,
            1.0 - residual_sum_squares / np.maximum(total_sum_squares, 1e-12),
            0.0,
        )
        r2 = float(np.mean(per_output_r2))
        return {"mse": mse, "mae": mae, "r2": r2}

    def evaluate_xy(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        return self.evaluate(X, Y)

    def save(self, path: str | Path) -> None:
        if self.weight is None or self.bias is None or self.x_mean is None or self.y_mean is None:
            raise ValueError("Cannot save an unfitted operator")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(
            path,
            weight=self.weight,
            bias=self.bias,
            x_mean=self.x_mean,
            y_mean=self.y_mean,
            ridge_lambda=np.asarray(self.config.ridge_lambda, dtype=np.float32),
            rank=np.asarray(-1 if self.config.rank is None else self.config.rank, dtype=np.int32),
            compute_backend=np.asarray(self.config.compute_backend),
        )

    @classmethod
    def load(cls, path: str | Path, *, name: str = "transport_operator") -> "TransportOperator":
        data = load_npz(path)
        rank = int(data["rank"])
        config = TransportOperatorConfig(
            ridge_lambda=float(data["ridge_lambda"]),
            rank=None if rank < 0 else rank,
            name=name,
            compute_backend=str(data["compute_backend"].item()) if "compute_backend" in data else "numpy",
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
            "compute_backend": self.config.compute_backend,
            "device": self.config.device,
            "train_metrics": self.train_metrics,
        }

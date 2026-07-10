from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0.0}
    if n_bootstrap <= 0:
        mean = float(np.mean(arr))
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": float(arr.size)}
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(int(random_seed))
    idx = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": float(arr.size),
    }


def add_bootstrap_ci(
    row: Dict[str, Any],
    *,
    metric_name: str,
    values: Sequence[float] | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> None:
    stats = bootstrap_mean_ci(
        values,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_seed=random_seed,
    )
    row[f"{metric_name}_ci_low"] = stats["ci_low"]
    row[f"{metric_name}_ci_high"] = stats["ci_high"]
    row[f"{metric_name}_n"] = stats["n"]


def paired_metric_deltas(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_policy: str,
    metric_keys: Sequence[str],
    policy_key: str = "policy_name",
    rank_key: str = "rank",
) -> list[Dict[str, Any]]:
    rows = list(rows)
    baseline_by_rank = {
        row.get(rank_key): row
        for row in rows
        if str(row.get(policy_key)) == baseline_policy
    }
    if not baseline_by_rank:
        raise ValueError(f"No baseline rows found for {baseline_policy}")

    default_baseline = next(iter(baseline_by_rank.values()))
    out: list[Dict[str, Any]] = []
    for row in rows:
        rank = row.get(rank_key)
        baseline = baseline_by_rank.get(rank, default_baseline)
        item: Dict[str, Any] = {
            policy_key: row.get(policy_key),
            rank_key: rank,
            "baseline_policy": baseline_policy,
        }
        for metric in metric_keys:
            if metric in row and metric in baseline:
                item[f"{metric}_delta"] = float(row[metric]) - float(baseline[metric])
        out.append(item)
    return out

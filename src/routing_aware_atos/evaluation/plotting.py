from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_policy_comparison(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path, *, metric: str = 'mean_r2') -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError('summary_rows cannot be empty')
    labels = [str(r['policy_name']) for r in rows]
    values = [float(r[metric]) for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.xlabel('policy')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_rank_sweep(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path, *, metric: str = 'mean_r2') -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError('summary_rows cannot be empty')

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row['policy_name']), []).append(row)

    plt.figure(figsize=(8, 4.5))
    for policy, items in grouped.items():
        items = sorted(items, key=lambda r: -1 if r['rank'] is None else int(r['rank']))
        x = [(-1 if r['rank'] is None else int(r['rank'])) for r in items]
        y = [float(r[metric]) for r in items]
        plt.plot(x, y, marker='o', label=policy)

    plt.xlabel('rank (-1 means full rank)')
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_causal_policy_comparison(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path, *, metric: str = 'feature_mse_restoration') -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError('summary_rows cannot be empty')
    labels = [str(r['policy_name']) for r in rows]
    values = [float(r[metric]) for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.xlabel('policy')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_causal_rank_sweep(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path, *, metric: str = 'feature_mse_restoration') -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError('summary_rows cannot be empty')

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row['policy_name']), []).append(row)

    plt.figure(figsize=(8, 4.5))
    for policy, items in grouped.items():
        items = sorted(items, key=lambda r: -1 if r['rank'] is None else int(r['rank']))
        x = [(-1 if r['rank'] is None else int(r['rank'])) for r in items]
        y = [float(r[metric]) for r in items]
        plt.plot(x, y, marker='o', label=policy)

    plt.xlabel('rank (-1 means full rank)')
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_transport_taxonomy_counts(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path) -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError("summary_rows cannot be empty")
    labels = [str(r["label"]) for r in rows]
    values = [float(r["count"]) for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("count")
    plt.xlabel("taxonomy label")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_transport_taxonomy_fractions(summary_rows: Iterable[Mapping[str, object]], output_path: str | Path) -> None:
    rows = list(summary_rows)
    if not rows:
        raise ValueError("summary_rows cannot be empty")
    labels = [str(r["label"]) for r in rows]
    values = [float(r["fraction"]) for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("fraction")
    plt.xlabel("taxonomy label")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

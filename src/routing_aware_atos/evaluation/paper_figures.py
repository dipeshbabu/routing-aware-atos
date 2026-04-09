from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from routing_aware_atos.evaluation.plotting import (
    plot_causal_policy_comparison,
    plot_causal_rank_sweep,
    plot_policy_comparison,
    plot_rank_sweep,
    plot_transport_taxonomy_counts,
    plot_transport_taxonomy_fractions,
)
from routing_aware_atos.utils.io import load_json


def generate_paper_figures(config: Mapping[str, Any]) -> Dict[str, str]:
    output_dir = Path(config.get("output_dir", "outputs/paper_figures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    if "policy_comparison_path" in config:
        payload = load_json(config["policy_comparison_path"])
        bar_metric = config.get("policy_bar_metric", "mean_r2")
        line_metric = config.get("policy_line_metric", "mean_r2")
        bar_path = output_dir / "figure_policy_comparison.png"
        line_path = output_dir / "figure_rank_sweep.png"
        plot_policy_comparison(payload["summary_rows"], bar_path, metric=bar_metric)
        plot_rank_sweep(payload["summary_rows"], line_path, metric=line_metric)
        written["policy_comparison"] = str(bar_path)
        written["rank_sweep"] = str(line_path)

    if "causal_comparison_path" in config:
        payload = load_json(config["causal_comparison_path"])
        bar_metric = config.get("causal_bar_metric", "feature_mse_restoration")
        line_metric = config.get("causal_line_metric", "feature_mse_restoration")
        bar_path = output_dir / "figure_causal_policy_comparison.png"
        line_path = output_dir / "figure_causal_rank_sweep.png"
        plot_causal_policy_comparison(payload["summary_rows"], bar_path, metric=bar_metric)
        plot_causal_rank_sweep(payload["summary_rows"], line_path, metric=line_metric)
        written["causal_policy_comparison"] = str(bar_path)
        written["causal_rank_sweep"] = str(line_path)

    if "taxonomy_path" in config:
        payload = load_json(config["taxonomy_path"])
        count_path = output_dir / "figure_taxonomy_counts.png"
        frac_path = output_dir / "figure_taxonomy_fractions.png"
        plot_transport_taxonomy_counts(payload["summary_rows"], count_path)
        plot_transport_taxonomy_fractions(payload["summary_rows"], frac_path)
        written["taxonomy_counts"] = str(count_path)
        written["taxonomy_fractions"] = str(frac_path)

    return written

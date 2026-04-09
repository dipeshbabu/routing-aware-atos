from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.causal_restore import compare_causal_policy_runs
from routing_aware_atos.evaluation.plotting import plot_causal_policy_comparison, plot_causal_rank_sweep
from routing_aware_atos.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark causal restoration across trained policy runs")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    payload = compare_causal_policy_runs(cfg["runs"], feature_ids=cfg.get("feature_ids"))

    output_dir = Path(cfg.get("output_dir", "outputs/causal_benchmarks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "causal_comparison.json", payload)

    bar_metric = cfg.get("bar_metric", "feature_mse_restoration")
    line_metric = cfg.get("line_metric", "feature_mse_restoration")
    plot_causal_policy_comparison(payload["summary_rows"], output_dir / "causal_policy_comparison.png", metric=bar_metric)
    plot_causal_rank_sweep(payload["summary_rows"], output_dir / "causal_rank_sweep.png", metric=line_metric)
    print(f"Saved causal benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()

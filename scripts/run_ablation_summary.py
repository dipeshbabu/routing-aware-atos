from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.analysis.ablations import compare_against_baseline, summarize_ablation_sweep
from routing_aware_atos.utils.io import load_json, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize policy or causal ablation sweeps")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    payload = load_json(cfg["input_path"])
    rows = payload[cfg.get("rows_key", "summary_rows")]

    summary = summarize_ablation_sweep(
        rows,
        group_key=cfg.get("group_key", "policy_name"),
        metric_keys=cfg.get("metric_keys", ["mean_r2"]),
    )
    save_json(cfg["summary_output_path"], summary)

    if "baseline_policy" in cfg:
        deltas = compare_against_baseline(
            rows,
            baseline_policy=cfg["baseline_policy"],
            policy_key=cfg.get("group_key", "policy_name"),
            metric_keys=cfg.get("metric_keys", ["mean_r2"]),
        )
        save_json(cfg["delta_output_path"], deltas)

    print(f"Saved ablation summary to {cfg['summary_output_path']}")


if __name__ == "__main__":
    main()

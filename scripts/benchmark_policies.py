from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.policy_comparison import compare_policy_runs, save_policy_comparison
from routing_aware_atos.evaluation.plotting import plot_policy_comparison, plot_rank_sweep
from routing_aware_atos.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark trained policy runs in feature space')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    payload = compare_policy_runs(cfg['runs'], feature_ids=cfg.get('feature_ids'))

    output_dir = Path(cfg.get('output_dir', 'outputs/benchmarks'))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_policy_comparison(payload, output_dir / 'comparison.json')
    plot_policy_comparison(payload['summary_rows'], output_dir / 'policy_comparison.png', metric=cfg.get('bar_metric', 'mean_r2'))
    plot_rank_sweep(payload['summary_rows'], output_dir / 'rank_sweep.png', metric=cfg.get('line_metric', 'mean_r2'))
    print(f'Saved benchmark outputs to {output_dir}')


if __name__ == '__main__':
    main()

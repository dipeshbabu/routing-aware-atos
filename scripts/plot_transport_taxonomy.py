from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.plotting import (
    plot_transport_taxonomy_counts,
    plot_transport_taxonomy_fractions,
)
from routing_aware_atos.utils.io import load_json, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved transport taxonomy payload")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    taxonomy = load_json(cfg["taxonomy_path"])
    output_dir = Path(cfg.get("output_dir", "outputs/taxonomy"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_transport_taxonomy_counts(taxonomy["summary_rows"], output_dir / "taxonomy_counts.png")
    plot_transport_taxonomy_fractions(taxonomy["summary_rows"], output_dir / "taxonomy_fractions.png")
    print(f"Saved taxonomy plots to {output_dir}")


if __name__ == "__main__":
    main()

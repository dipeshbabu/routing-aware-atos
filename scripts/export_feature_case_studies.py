from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.utils.io import load_json, load_yaml, save_json


def _best_examples(feature_rows: list[dict], label: str, k: int = 5) -> list[dict]:
    rows = [row for row in feature_rows if row["label"] == label]
    rows = sorted(rows, key=lambda x: x["best_r2"], reverse=True)
    return rows[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feature case studies from a saved taxonomy payload")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    taxonomy_path_value = cfg.get("taxonomy_path")
    if taxonomy_path_value is None:
        results_dir = cfg.get("results_dir")
        if results_dir is None:
            raise ValueError("Config must define `taxonomy_path` or `results_dir`.")
        taxonomy_path = Path(results_dir) / "transport_taxonomy.json"
    else:
        taxonomy_path = Path(taxonomy_path_value)
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Missing taxonomy file: {taxonomy_path}. Run build_transport_taxonomy.py first."
        )

    taxonomy = load_json(taxonomy_path)
    feature_rows = taxonomy["feature_rows"]

    case_studies = {
        "same_token_transport": _best_examples(feature_rows, "same_token_transport"),
        "cross_token_transport": _best_examples(feature_rows, "cross_token_transport"),
        "multi_source_transport": _best_examples(feature_rows, "multi_source_transport"),
        "unexplained_or_synthesized": _best_examples(
            feature_rows,
            "unexplained_or_synthesized",
        ),
    }

    out_path = Path(
        cfg.get(
            "output_path",
            taxonomy_path.with_name("feature_case_studies.json"),
        )
    )
    save_json(out_path, case_studies)
    print(f"Saved case studies to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.evaluation.paper_figures import generate_paper_figures
from routing_aware_atos.utils.io import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-facing figures from saved payloads")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    written = generate_paper_figures(cfg)
    manifest_path = Path(cfg.get("output_dir", "outputs/paper_figures")) / "manifest.json"
    save_json(manifest_path, written)
    print(f"Saved paper figures manifest to {manifest_path}")


if __name__ == "__main__":
    main()

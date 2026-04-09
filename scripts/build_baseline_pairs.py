from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.data.baseline_pairs import SameTokenBaselineBuilder
from routing_aware_atos.data.mock_cache import make_mock_samples
from routing_aware_atos.utils.io import load_cached_samples, load_yaml, save_json, save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-token baseline training pairs.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if cfg.get("cache_path"):
        samples = load_cached_samples(cfg["cache_path"])
    else:
        samples = make_mock_samples(
            num_samples=cfg.get("num_samples", 2),
            seq_len=cfg.get("seq_len", 6),
            d_model=cfg.get("d_model", 4),
        )

    builder = SameTokenBaselineBuilder(
        samples=samples,
        source_layer=cfg["source_layer"],
        target_layer=cfg["target_layer"],
        include_positions=cfg.get("include_positions"),
    )
    X, Y, metadata = builder.build_pairs()

    out_dir = Path(cfg.get("output_dir", "outputs/baseline"))
    save_npz(out_dir / "pairs.npz", X=X, Y=Y)
    save_json(out_dir / "metadata.json", metadata)

    print(f"Saved baseline pairs: X{X.shape}, Y{Y.shape} -> {out_dir}")


if __name__ == "__main__":
    main()

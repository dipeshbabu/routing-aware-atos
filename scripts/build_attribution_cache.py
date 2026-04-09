from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from routing_aware_atos.data.attribution_cache import AttributionBuildConfig, attach_attribution_scores, summarize_attribution_scores
from routing_aware_atos.utils.io import load_cached_samples, load_yaml, save_cached_samples, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build attribution score caches from residuals and attention")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    samples = load_cached_samples(cfg["samples_path"])
    layer_pairs = [tuple(x) for x in cfg["layer_pairs"]]

    build_cfg = AttributionBuildConfig(
        methods=cfg.get("methods", ["attention_value"]),
        normalize_rows=cfg.get("normalize_rows", True),
        symmetrize_similarity=cfg.get("symmetrize_similarity", False),
        include_existing=cfg.get("include_existing", True),
    )
    updated = attach_attribution_scores(samples, layer_pairs, config=build_cfg)
    output_path = cfg["output_path"]
    save_cached_samples(output_path, updated)

    summary = {
        f"{src}->{tgt}": summarize_attribution_scores(updated, (src, tgt))
        for src, tgt in layer_pairs
    }
    summary_path = cfg.get("summary_path")
    if summary_path:
        save_json(summary_path, summary)

    print(f"Saved attribution-enriched samples to {output_path}")


if __name__ == "__main__":
    main()

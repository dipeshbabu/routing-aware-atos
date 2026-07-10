from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

import numpy as np

from routing_aware_atos.evaluation.transport_efficiency import evaluate_operator_transport_efficiency
from routing_aware_atos.models.transport_operator import TransportOperator
from routing_aware_atos.utils.io import load_npz, load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ATO transport efficiency and CCA ceiling.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    operator = TransportOperator.load(cfg["operator_path"])
    pairs = load_npz(cfg["pairs_path"])
    X = np.asarray(pairs["X"], dtype=np.float32)
    Y = np.asarray(pairs["Y"], dtype=np.float32)

    metrics = evaluate_operator_transport_efficiency(
        operator,
        X,
        Y,
        rank=cfg.get("rank"),
        eps=cfg.get("eps", 1e-8),
    )
    payload = {
        "operator_path": cfg["operator_path"],
        "pairs_path": cfg["pairs_path"],
        "metrics": metrics.to_dict(),
    }
    if "output_path" in cfg:
        save_json(cfg["output_path"], payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()

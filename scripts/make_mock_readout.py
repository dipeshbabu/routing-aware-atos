from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np

from routing_aware_atos.utils.io import save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mock readout matrix for logit restoration demos")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    readout = rng.normal(size=(args.d_model, args.vocab_size)).astype(np.float32)
    save_npz(args.output, readout=readout)
    print(f"Saved mock readout to {args.output}")


if __name__ == "__main__":
    main()

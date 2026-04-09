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
    parser = argparse.ArgumentParser(description='Create a mock SAE decoder for tests and demos')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=8)
    parser.add_argument('--d-model', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    decoder = rng.normal(size=(args.num_features, args.d_model)).astype(np.float32)
    save_npz(args.output, decoder=decoder)
    print(f'Saved decoder to {args.output}')


if __name__ == '__main__':
    main()
